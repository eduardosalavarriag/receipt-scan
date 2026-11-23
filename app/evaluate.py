"""The ``evaluate`` command: score the current extractor on a data folder.

Run as ``docker run -v ./data:/data <image> evaluate``. It reports, for both the
store and the total:

* **accuracy** — store allows small OCR typos, total is exact to the cent;
* a **per-source breakdown** (``sroie`` / ``cord`` / ``synthetic`` / ``labelled``);
* **retakes folded by majority vote** — several photos of one physical receipt
  (sharing a ``receipt_id``) count as a single receipt, voting on the answer;
* an **OCR-miss vs picker-miss split** of the failures: whether the right answer
  was never among the OCR words at all (OCR miss) or was there but the extractor
  chose wrong (picker miss).

The extractor scored here is whatever the service uses: the simple rules from
:mod:`app.extract` today, reading the stored OCR words of each example. Run first
on the simple rules, it yields the baseline the accuracy guard (#10) protects.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from .dataset import Label, Word, iter_labels
from .extract import extract_fields, parse_amount

# A predicted store counts as correct if its normalised edit distance to the gold
# store is within this fraction of the longer string (small OCR typos allowed).
_STORE_TYPO_RATIO = 0.2


# ---------------------------------------------------------------------------
# Turning stored words back into the text the extractor reads
# ---------------------------------------------------------------------------
@dataclass
class _Row:
    centre: float
    height: float
    items: list[tuple[int, str]]


def reconstruct_text(words: list[Word]) -> str:
    """Lay stored words back out as text: group into rows, top-to-bottom.

    Words whose vertical centres sit close together form one line (ordered left
    to right); lines are ordered top to bottom. This is the text the current
    text-based extractor consumes, standing in for live OCR output.
    """
    rows: list[_Row] = []
    for word in sorted(words, key=lambda w: ((w["box"][1] + w["box"][3]) / 2, w["box"][0])):
        x0, y0, _x1, y1 = word["box"]
        centre = (y0 + y1) / 2
        height = max(y1 - y0, 1)
        placed = False
        for row in rows:
            if abs(row.centre - centre) <= 0.6 * max(height, row.height):
                row.items.append((x0, word["text"]))
                row.height = max(row.height, height)
                placed = True
                break
        if not placed:
            rows.append(_Row(centre=centre, height=height, items=[(x0, word["text"])]))

    rows.sort(key=lambda r: r.centre)
    lines = []
    for row in rows:
        row.items.sort(key=lambda it: it[0])
        lines.append(" ".join(text for _x, text in row.items))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Matching gold answers against predictions
# ---------------------------------------------------------------------------
def _normalise_store(value: str) -> str:
    return " ".join(value.upper().split())


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def store_matches(predicted: str | None, gold: str | None) -> bool:
    """True when the predicted store equals the gold store, small typos aside."""
    if gold is None or predicted is None:
        return gold is None and predicted is None
    pred, want = _normalise_store(predicted), _normalise_store(gold)
    if not want:
        return not pred
    if pred == want:
        return True
    allowed = max(1, round(_STORE_TYPO_RATIO * max(len(pred), len(want))))
    return _edit_distance(pred, want) <= allowed


def _to_cents(value: str | None) -> Decimal | None:
    if value is None:
        return None
    amount = parse_amount(value)
    return None if amount is None else amount.quantize(Decimal("0.01"))


def total_matches(predicted: str | None, gold: str | None) -> bool:
    """True when the predicted total equals the gold total, exact to the cent."""
    p, g = _to_cents(predicted), _to_cents(gold)
    if g is None or p is None:
        return g is None and p is None
    return p == g


# ---------------------------------------------------------------------------
# Was the gold answer even among the OCR words? (OCR-miss vs picker-miss)
# ---------------------------------------------------------------------------
def _store_in_words(words: list[Word], gold: str | None) -> bool:
    if gold is None:
        return False
    for line in reconstruct_text(words).splitlines():
        if store_matches(line, gold):
            return True
    return any(store_matches(w["text"], gold) for w in words)


def _total_in_words(words: list[Word], gold: str | None) -> bool:
    want = _to_cents(gold)
    if want is None:
        return False
    for word in words:
        for token in word["text"].split():
            amount = parse_amount(token)
            if amount is not None and amount.quantize(Decimal("0.01")) == want:
                return True
    return False


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------
@dataclass
class FieldStats:
    """Tallies for one field (store or total) over a set of receipts."""

    total: int = 0
    correct: int = 0
    ocr_miss: int = 0
    picker_miss: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def record(self, correct: bool, answer_in_words: bool) -> None:
        self.total += 1
        if correct:
            self.correct += 1
        elif answer_in_words:
            self.picker_miss += 1
        else:
            self.ocr_miss += 1


@dataclass
class SourceStats:
    source: str
    store: FieldStats = field(default_factory=FieldStats)
    total: FieldStats = field(default_factory=FieldStats)


@dataclass
class EvaluationResult:
    groups: int = 0
    examples: int = 0
    store: FieldStats = field(default_factory=FieldStats)
    total: FieldStats = field(default_factory=FieldStats)
    per_source: list[SourceStats] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Running the extractor and voting
# ---------------------------------------------------------------------------
def _predict(words: list[Word]) -> tuple[str | None, str | None]:
    fields = extract_fields(reconstruct_text(words))
    store = fields["store"]
    total = fields["total"]
    assert isinstance(store, dict) and isinstance(total, dict)
    store_value = store["value"]
    total_value = total["value"]
    return (
        store_value if isinstance(store_value, str) else None,
        total_value if isinstance(total_value, str) else None,
    )


def _majority(values: list[str | None]) -> str | None:
    """Most common prediction, ties broken toward the first one seen."""
    counts = Counter(values)
    best = max(counts, key=lambda v: (counts[v], -values.index(v)))
    return best


def _group_key(example_id: str, label: Label) -> str:
    receipt_id = label.get("receipt_id")
    return receipt_id if receipt_id else example_id


def evaluate_data_dir(data_dir: Path | str) -> EvaluationResult:
    """Score the current extractor over every labelled example in ``data_dir``."""
    groups: dict[str, list[Label]] = {}
    order: list[str] = []
    examples = 0
    for example_id, label in iter_labels(data_dir):
        examples += 1
        key = _group_key(example_id, label)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(label)

    result = EvaluationResult(groups=len(groups), examples=examples)
    sources: dict[str, SourceStats] = {}

    for key in order:
        members = groups[key]
        source = Counter(m["source"] for m in members).most_common(1)[0][0]
        gold_store = Counter(m.get("company") for m in members).most_common(1)[0][0]
        gold_total = Counter(m.get("total") for m in members).most_common(1)[0][0]

        predictions = [_predict(m["words"]) for m in members]
        pred_store = _majority([p[0] for p in predictions])
        pred_total = _majority([p[1] for p in predictions])

        all_words: list[Word] = [w for m in members for w in m["words"]]
        store_ok = store_matches(pred_store, gold_store)
        total_ok = total_matches(pred_total, gold_total)
        store_present = _store_in_words(all_words, gold_store)
        total_present = _total_in_words(all_words, gold_total)

        result.store.record(store_ok, store_present)
        result.total.record(total_ok, total_present)

        src = sources.setdefault(source, SourceStats(source))
        src.store.record(store_ok, store_present)
        src.total.record(total_ok, total_present)

    result.per_source = [sources[s] for s in sorted(sources)]
    return result


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------
def _pct(stats: FieldStats) -> str:
    return f"{stats.accuracy * 100:.1f}% ({stats.correct}/{stats.total})"


def _misses(stats: FieldStats) -> str:
    return f"{stats.ocr_miss} OCR-miss, {stats.picker_miss} picker-miss"


def format_report(result: EvaluationResult) -> str:
    lines = [
        f"Evaluated {result.groups} receipt(s) from {result.examples} example(s).",
        "",
        f"Store accuracy: {_pct(result.store)}",
        f"Total accuracy: {_pct(result.total)}",
    ]
    if result.per_source:
        lines.append("")
        lines.append("Per source:")
        width = max(len(s.source) for s in result.per_source)
        for src in result.per_source:
            lines.append(
                f"  {src.source.ljust(width)}  "
                f"store {_pct(src.store)}   total {_pct(src.total)}"
            )
    lines.append("")
    lines.append("Misses:")
    lines.append(f"  store: {_misses(result.store)}")
    lines.append(f"  total: {_misses(result.total)}")
    return "\n".join(lines)
