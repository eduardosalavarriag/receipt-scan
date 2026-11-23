from __future__ import annotations

from pathlib import Path

from app.dataset import Label, write_example
from app.evaluate import (
    evaluate_data_dir,
    format_report,
    reconstruct_text,
    store_matches,
)


def _label(
    *,
    source: str,
    company: str | None,
    total: str | None,
    words: list[tuple[str, list[int]]],
    receipt_id: str | None = None,
) -> Label:
    label: Label = {
        "source": source,
        "company": company,
        "total": total,
        "words": [{"text": t, "box": b} for t, b in words],
    }
    if receipt_id is not None:
        label["receipt_id"] = receipt_id
    return label


def _good_receipt(company: str, total: str, source: str = "synthetic") -> Label:
    """A receipt whose top line is the store and whose total is keyword-tagged."""
    return _label(
        source=source,
        company=company,
        total=total,
        words=[
            (company, [10, 10, 200, 40]),
            ("Milch 1,00 €", [10, 80, 200, 110]),
            (f"SUMME {total.replace('.', ',')} €", [10, 150, 200, 180]),
        ],
    )


# ---------------------------------------------------------------------------
# reconstruct_text
# ---------------------------------------------------------------------------
def test_reconstruct_text_groups_rows_top_to_bottom() -> None:
    words = [
        {"text": "WORLD", "box": [120, 10, 200, 40]},
        {"text": "HELLO", "box": [10, 12, 100, 38]},
        {"text": "bottom", "box": [10, 200, 100, 230]},
    ]
    assert reconstruct_text(words) == "HELLO WORLD\nbottom"


# ---------------------------------------------------------------------------
# store fuzzy matching
# ---------------------------------------------------------------------------
def test_store_matches_allows_small_typos() -> None:
    assert store_matches("TESCO STORE", "TESC0 STORE") is True
    assert store_matches("TESCO", "ALDI") is False
    assert store_matches(None, None) is True
    assert store_matches("TESCO", None) is False


# ---------------------------------------------------------------------------
# overall accuracy + per-source breakdown
# ---------------------------------------------------------------------------
def test_evaluate_reports_accuracy_and_breakdown(tmp_path: Path) -> None:
    write_example(tmp_path, "s1", b"img", _good_receipt("ALDI SUED", "12.50", "sroie"))
    write_example(tmp_path, "c1", b"img", _good_receipt("REWE MARKT", "4.99", "cord"))

    result = evaluate_data_dir(tmp_path)

    assert result.groups == 2
    assert result.store.correct == 2
    assert result.total.correct == 2
    assert result.store.accuracy == 1.0
    assert result.total.accuracy == 1.0
    assert {s.source for s in result.per_source} == {"sroie", "cord"}

    report = format_report(result)
    assert "Store accuracy" in report
    assert "Total accuracy" in report
    assert "sroie" in report and "cord" in report


# ---------------------------------------------------------------------------
# retakes: majority vote, counted once
# ---------------------------------------------------------------------------
def test_retakes_count_once_by_majority_vote(tmp_path: Path) -> None:
    # Three photos of the same receipt. Two read the total correctly; one is a
    # blurry retake whose total word is garbled. Majority wins, counts as one.
    write_example(tmp_path, "r-a", b"img", _good_receipt("EDEKA", "19.90"))
    write_example(tmp_path, "r-b", b"img", _good_receipt("EDEKA", "19.90"))
    bad = _good_receipt("EDEKA", "19.90")
    # Corrupt the total line so this single retake reads a different amount.
    bad["words"][2] = {"text": "SUMME 88,88 €", "box": [10, 150, 200, 180]}
    write_example(tmp_path, "r-c", b"img", bad)
    # Tie every example to the same receipt.
    for ident in ("r-a", "r-b", "r-c"):
        label_path = tmp_path / "labels" / f"{ident}.json"
        import json

        data = json.loads(label_path.read_text())
        data["receipt_id"] = "receipt-1"
        label_path.write_text(json.dumps(data))

    result = evaluate_data_dir(tmp_path)

    assert result.groups == 1
    assert result.examples == 3
    assert result.total.correct == 1  # majority vote recovered the right total


# ---------------------------------------------------------------------------
# OCR-miss vs picker-miss split
# ---------------------------------------------------------------------------
def test_ocr_miss_vs_picker_miss_split(tmp_path: Path) -> None:
    # Picker miss: the correct total (2,00) is present in the words, but a larger
    # non-total amount (99,00) with no keyword lures the fallback "max amount".
    picker = _label(
        source="synthetic",
        company="SHOP",
        total="2.00",
        words=[
            ("SHOP", [10, 10, 200, 40]),
            ("Deposit 99,00 €", [10, 80, 200, 110]),
            ("Paid 2,00 €", [10, 150, 200, 180]),
        ],
    )
    write_example(tmp_path, "p1", b"img", picker)

    # OCR miss: the gold total never appears among the OCR words at all.
    ocr = _label(
        source="synthetic",
        company="SHOP",
        total="77.77",
        words=[
            ("SHOP", [10, 10, 200, 40]),
            ("Milch 1,00 €", [10, 80, 200, 110]),
        ],
    )
    write_example(tmp_path, "o1", b"img", ocr)

    result = evaluate_data_dir(tmp_path)

    assert result.total.correct == 0
    assert result.total.picker_miss == 1
    assert result.total.ocr_miss == 1

    report = format_report(result)
    assert "OCR-miss" in report
    assert "picker-miss" in report
