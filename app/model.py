"""The ML pickers: a store picker and a total picker, each an ``MLPClassifier``.

Each picker scores every OCR candidate on a receipt and the highest score wins.
A candidate's features are:

* its **text**, hashed with :class:`~sklearn.feature_extraction.text.HashingVectorizer`
  (character n-grams). Hashing keeps **no vocabulary** on disk, so a trained model
  never carries the words it was trained on;
* **position and shape** — where the line sits on the receipt and how big it is;
* for totals, whether the line reads like a grand-total line (keyword, currency)
  and how the amount compares to the largest amount on the receipt.

The picker's probability for the winning candidate becomes the answer's
confidence. ``train`` (see :mod:`app.train`) fits the two classifiers and writes
them to ``model.joblib``; the service and ``evaluate`` load that file when present.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import HashingVectorizer

from .dataset import Word
from .evaluate import group_lines
from .extract import (
    ExtractedFields,
    StoreField,
    TotalField,
    amounts_in,
    detect_currency,
    format_amount,
)

# Where the trained model lives inside the data directory, and the optional
# built-in model shipped in the image (added by the release build, #9).
MODEL_FILENAME = "model.joblib"
BUILTIN_MODEL_PATH = Path(__file__).resolve().parent / "builtin_model.joblib"

# Hashing dimensionality for the text features. Large enough to keep character
# n-gram collisions rare, small enough to train in seconds.
_TEXT_FEATURES = 2**12

# Keywords that mark a grand-total line, as a position/shape feature. The text
# hasher also learns keywords on its own, so this only speeds learning for the
# common ones; novel keywords (e.g. "BETRAG") are picked up from the text.
_TOTAL_KEYWORDS = ("ZU ZAHLEN", "SUMME", "GESAMT", "TOTAL", "BETRAG")


def make_vectorizer() -> HashingVectorizer:
    """A stateless, vocabulary-free text hasher (identical at fit and predict)."""
    return HashingVectorizer(
        n_features=_TEXT_FEATURES,
        analyzer="char_wb",
        ngram_range=(2, 4),
        alternate_sign=False,
        norm="l2",
    )


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
@dataclass
class StoreCandidate:
    text: str
    box: list[int]


@dataclass
class TotalCandidate:
    amount: Decimal
    line_text: str
    box: list[int]


@dataclass(frozen=True)
class _Bounds:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(self.x1 - self.x0, 1.0)

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1.0)


def _bounds(words: list[Word]) -> _Bounds:
    if not words:
        return _Bounds(0.0, 0.0, 1.0, 1.0)
    xs0 = min(w["box"][0] for w in words)
    ys0 = min(w["box"][1] for w in words)
    xs1 = max(w["box"][2] for w in words)
    ys1 = max(w["box"][3] for w in words)
    return _Bounds(float(xs0), float(ys0), float(xs1), float(ys1))


def store_candidates(words: list[Word]) -> list[StoreCandidate]:
    """Every text line is a candidate store name."""
    return [StoreCandidate(text=text, box=box) for text, box in group_lines(words)]


def total_candidates(words: list[Word]) -> list[TotalCandidate]:
    """Every parseable amount on every line is a candidate total."""
    candidates: list[TotalCandidate] = []
    for text, box in group_lines(words):
        for amount in amounts_in(text):
            candidates.append(TotalCandidate(amount=amount, line_text=text, box=box))
    return candidates


# ---------------------------------------------------------------------------
# Numeric (position / shape) features
# ---------------------------------------------------------------------------
def _position(box: list[int], bounds: _Bounds) -> tuple[float, float, float]:
    x0, y0, x1, y1 = box
    cx = ((x0 + x1) / 2 - bounds.x0) / bounds.width
    cy = ((y0 + y1) / 2 - bounds.y0) / bounds.height
    rel_h = (y1 - y0) / bounds.height
    return cx, cy, rel_h


def _store_numeric(cand: StoreCandidate, bounds: _Bounds) -> list[float]:
    cx, cy, rel_h = _position(cand.box, bounds)
    text_len = min(len(cand.text) / 40.0, 2.0)
    return [cx, cy, rel_h, text_len]


def _total_numeric(
    cand: TotalCandidate, bounds: _Bounds, max_amount: Decimal
) -> list[float]:
    cx, cy, rel_h = _position(cand.box, bounds)
    upper = cand.line_text.upper()
    has_keyword = 1.0 if any(kw in upper for kw in _TOTAL_KEYWORDS) else 0.0
    has_currency = 1.0 if detect_currency(cand.line_text) else 0.0
    amount_frac = float(cand.amount) / float(max_amount) if max_amount else 0.0
    is_max = 1.0 if cand.amount == max_amount else 0.0
    return [cx, cy, rel_h, has_keyword, has_currency, amount_frac, is_max]


def _store_features(
    vectorizer: HashingVectorizer, cands: list[StoreCandidate], bounds: _Bounds
) -> csr_matrix:
    text = vectorizer.transform([c.text for c in cands])
    numeric = csr_matrix(np.array([_store_numeric(c, bounds) for c in cands], dtype=float))
    return hstack([text, numeric]).tocsr()


def _total_features(
    vectorizer: HashingVectorizer, cands: list[TotalCandidate], bounds: _Bounds
) -> csr_matrix:
    max_amount = max((c.amount for c in cands), default=Decimal(0))
    text = vectorizer.transform([c.line_text for c in cands])
    numeric = csr_matrix(
        np.array([_total_numeric(c, bounds, max_amount) for c in cands], dtype=float)
    )
    return hstack([text, numeric]).tocsr()


def _positive_scores(clf: ClassifierMixin, features: csr_matrix) -> np.ndarray:
    """Probability of the positive class (1) for each row, 0.0 if never seen."""
    proba = clf.predict_proba(features)
    classes = list(clf.classes_)
    if 1 not in classes:
        return np.zeros(features.shape[0])
    return proba[:, classes.index(1)]


# ---------------------------------------------------------------------------
# The trained model
# ---------------------------------------------------------------------------
@dataclass
class Model:
    vectorizer: HashingVectorizer
    store_clf: ClassifierMixin
    total_clf: ClassifierMixin

    def predict_store(self, words: list[Word]) -> tuple[str | None, float]:
        cands = store_candidates(words)
        if not cands:
            return None, 0.0
        features = _store_features(self.vectorizer, cands, _bounds(words))
        scores = _positive_scores(self.store_clf, features)
        best = int(np.argmax(scores))
        return cands[best].text, float(scores[best])

    def predict_total(self, words: list[Word]) -> tuple[str | None, str | None, float]:
        cands = total_candidates(words)
        if not cands:
            return None, None, 0.0
        features = _total_features(self.vectorizer, cands, _bounds(words))
        scores = _positive_scores(self.total_clf, features)
        best = int(np.argmax(scores))
        chosen = cands[best]
        currency = detect_currency(chosen.line_text) or detect_currency(
            "\n".join(c.line_text for c in cands)
        )
        return format_amount(chosen.amount), currency, float(scores[best])

    def predict(self, words: list[Word]) -> tuple[str | None, str | None]:
        """A :data:`~app.evaluate.Predictor`: ``(store, total)`` without scores."""
        store, _ = self.predict_store(words)
        total, _currency, _ = self.predict_total(words)
        return store, total

    def extract_fields(self, words: list[Word]) -> ExtractedFields:
        """Store and total fields in the same shape as the rules extractor."""
        store_value, store_conf = self.predict_store(words)
        total_value, currency, total_conf = self.predict_total(words)
        store_field: StoreField = {
            "value": store_value,
            "confidence": store_conf if store_value is not None else 0.0,
        }
        if total_value is not None:
            total_field: TotalField = {
                "value": total_value,
                "currency": currency,
                "confidence": total_conf,
            }
        else:
            total_field = {"value": None, "currency": None, "confidence": 0.0}
        return {"store": store_field, "total": total_field}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def model_file(data_dir: Path | str) -> Path:
    return Path(data_dir) / MODEL_FILENAME


def find_model_path(data_dir: Path | str) -> Path | None:
    """The trained model to use: the one in ``data_dir``, else the built-in one."""
    candidate = model_file(data_dir)
    if candidate.exists():
        return candidate
    if BUILTIN_MODEL_PATH.exists():
        return BUILTIN_MODEL_PATH
    return None


def save_model(model: Model, path: Path | str) -> None:
    joblib.dump(
        {
            "version": 1,
            "vectorizer": model.vectorizer,
            "store_clf": model.store_clf,
            "total_clf": model.total_clf,
        },
        path,
    )


def load_model(path: Path | str) -> Model:
    payload = joblib.load(path)
    return Model(
        vectorizer=payload["vectorizer"],
        store_clf=payload["store_clf"],
        total_clf=payload["total_clf"],
    )


# A tiny cache so the service does not re-read the model from disk on every
# request. Keyed by path and modification time, so a freshly trained model is
# picked up automatically.
_cache: dict[Path, tuple[float, Model]] = {}


def load_cached_model(data_dir: Path | str) -> Model | None:
    """Load the active model for ``data_dir``, cached by file modification time."""
    path = find_model_path(data_dir)
    if path is None:
        return None
    mtime = path.stat().st_mtime
    cached = _cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    model = load_model(path)
    _cache[path] = (mtime, model)
    return model
