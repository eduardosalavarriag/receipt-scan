"""The ``train`` command: fit the ML pickers from everything in the data folder.

Run as ``docker run -v ./data:/data <image> train``. It reads every labelled
example (see :mod:`app.dataset`), builds one training row per OCR candidate,
holds out a slice of the *receipts* (never splitting a receipt's retakes across
the divide), fits a store picker and a total picker, scores the held-out slice
through :mod:`app.evaluate`, and writes ``model.joblib`` into the data folder.

The classifiers are small :class:`~sklearn.neural_network.MLPClassifier` networks
over hashed text plus position/shape features (see :mod:`app.model`), so training
the full downloaded dataset finishes in minutes on a normal CPU.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier

from .dataset import Label, iter_labels
from .evaluate import (
    EvaluationResult,
    evaluate_examples,
    group_key,
    store_matches,
    total_matches,
)
from .extract import format_amount
from .model import (
    _TEXT_FEATURES,
    Model,
    _bounds,
    _store_features,
    _total_features,
    make_vectorizer,
    model_file,
    save_model,
    store_candidates,
    total_candidates,
)

# Fraction of receipts held out to score the trained model. At least one receipt
# is always held out when there is more than one.
DEFAULT_HOLDOUT_FRACTION = 0.2


@dataclass
class TrainResult:
    train_receipts: int
    holdout_receipts: int
    store_rows: int
    total_rows: int
    evaluation: EvaluationResult
    model_path: Path


def _split_by_receipt(
    examples: list[tuple[str, Label]], fraction: float, seed: int
) -> tuple[list[tuple[str, Label]], list[tuple[str, Label]]]:
    """Split examples into (train, holdout) without separating a receipt's retakes."""
    # Unique receipt keys in first-seen order (dict preserves insertion order).
    keys = list(dict.fromkeys(group_key(i, lab) for i, lab in examples))
    rng = random.Random(seed)
    shuffled = keys[:]
    rng.shuffle(shuffled)

    n_holdout = 0
    if len(shuffled) > 1:
        n_holdout = max(1, round(len(shuffled) * fraction))
        n_holdout = min(n_holdout, len(shuffled) - 1)
    holdout_keys = set(shuffled[:n_holdout])

    train = [(i, lab) for i, lab in examples if group_key(i, lab) not in holdout_keys]
    holdout = [(i, lab) for i, lab in examples if group_key(i, lab) in holdout_keys]
    return train, holdout


def _fit_classifier(features: csr_matrix, targets: np.ndarray) -> object:
    """Fit an MLP, or a constant classifier when there is only one (or no) class."""
    if features.shape[0] == 0:
        # No candidates of this kind anywhere: a classifier that always abstains.
        return DummyClassifier(strategy="constant", constant=0).fit(
            csr_matrix((1, features.shape[1])), np.array([0])
        )
    if len(np.unique(targets)) < 2:
        return DummyClassifier(strategy="prior").fit(features, targets)
    clf = MLPClassifier(
        hidden_layer_sizes=(32,),
        max_iter=400,
        random_state=0,
    )
    clf.fit(features, targets)
    return clf


def fit_model(examples: list[tuple[str, Label]]) -> tuple[Model, int, int]:
    """Fit the store and total pickers from labelled examples.

    Returns the model and the number of store and total training rows used.
    """
    vectorizer = make_vectorizer()

    store_blocks: list[csr_matrix] = []
    store_targets: list[int] = []
    total_blocks: list[csr_matrix] = []
    total_targets: list[int] = []

    for _example_id, label in examples:
        words = label["words"]
        bounds = _bounds(words)
        gold_store = label.get("company")
        gold_total = label.get("total")

        store_cands = store_candidates(words)
        if store_cands:
            store_blocks.append(_store_features(vectorizer, store_cands, bounds))
            store_targets.extend(
                1 if store_matches(c.text, gold_store) else 0 for c in store_cands
            )

        total_cands = total_candidates(words)
        if total_cands:
            total_blocks.append(_total_features(vectorizer, total_cands, bounds))
            total_targets.extend(
                1 if total_matches(format_amount(c.amount), gold_total) else 0
                for c in total_cands
            )

    store_clf = _fit_classifier(_stack(store_blocks), np.array(store_targets))
    total_clf = _fit_classifier(_stack(total_blocks), np.array(total_targets))
    model = Model(vectorizer=vectorizer, store_clf=store_clf, total_clf=total_clf)
    return model, len(store_targets), len(total_targets)


def _stack(blocks: list[csr_matrix]) -> csr_matrix:
    if not blocks:
        # No candidates of this kind anywhere (degenerate dataset); the matching
        # classifier abstains regardless of width.
        return csr_matrix((0, _TEXT_FEATURES + 1))
    return vstack(blocks).tocsr()


def train_model(
    data_dir: Path | str,
    *,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    seed: int = 0,
) -> TrainResult:
    """Fit the pickers on ``data_dir``, score the held-out receipts, save the model."""
    examples = list(iter_labels(data_dir))
    if not examples:
        raise ValueError(f"No labelled examples found in {data_dir}.")

    train, holdout = _split_by_receipt(examples, holdout_fraction, seed)
    model, store_rows, total_rows = fit_model(train)

    scored = holdout if holdout else train
    evaluation = evaluate_examples(scored, predict=model.predict)

    path = model_file(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, path)

    train_keys = {group_key(i, lab) for i, lab in train}
    holdout_keys = {group_key(i, lab) for i, lab in holdout}
    return TrainResult(
        train_receipts=len(train_keys),
        holdout_receipts=len(holdout_keys),
        store_rows=store_rows,
        total_rows=total_rows,
        evaluation=evaluation,
        model_path=path,
    )
