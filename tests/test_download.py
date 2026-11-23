from __future__ import annotations

import io
from decimal import Decimal
from pathlib import Path

from PIL import Image

from app.dataset import Label, example_exists, iter_labels, write_example
from app.download import (
    convert_cord,
    convert_sroie,
    format_de_amount,
    format_de_integer,
    generate_german_from_label,
    generate_german_receipts,
    parse_sroie_box_file,
    render_words_image,
)
from app.extract import parse_amount


def test_format_de_amount() -> None:
    assert format_de_amount(Decimal("1234.5")) == "1.234,50"
    assert format_de_amount(Decimal("51.76")) == "51,76"
    assert format_de_amount(Decimal("9")) == "9,00"


def test_format_de_integer() -> None:
    assert format_de_integer(Decimal("60000")) == "60.000"
    assert format_de_integer(Decimal("7")) == "7"


# ---------------------------------------------------------------------------
# SROIE / CORD conversion
# ---------------------------------------------------------------------------
_SROIE_BOX = (
    "10,10,100,10,100,30,10,30,TESCO STORE\n"
    "10,200,120,200,120,220,10,220,TOTAL 9.00\n"
)


def test_parse_sroie_box_file() -> None:
    words = parse_sroie_box_file(_SROIE_BOX)
    assert words[0] == {"text": "TESCO STORE", "box": [10, 10, 100, 30]}
    assert words[1]["text"] == "TOTAL 9.00"


def test_convert_sroie() -> None:
    label = convert_sroie(_SROIE_BOX, {"company": "TESCO STORE", "total": "9.00"})
    assert label["source"] == "sroie"
    assert label["company"] == "TESCO STORE"
    assert label["total"] == "9.00"
    assert len(label["words"]) == 2


def test_convert_cord() -> None:
    annotation = {
        "valid_line": [
            {
                "category": "menu.nm",
                "words": [{"quad": {"x1": 10, "y1": 10, "x2": 90, "y2": 10,
                                    "x3": 90, "y3": 30, "x4": 10, "y4": 30},
                           "text": "Cafe"}],
            },
            {
                "category": "total.total_price",
                "words": [{"quad": {"x1": 10, "y1": 200, "x2": 80, "y2": 200,
                                    "x3": 80, "y3": 220, "x4": 10, "y4": 220},
                           "text": "12.50"}],
            },
        ]
    }
    label = convert_cord(annotation)
    assert label["source"] == "cord"
    assert label["company"] == "Cafe"
    assert label["total"] == "12.50"
    assert len(label["words"]) == 2


# ---------------------------------------------------------------------------
# German generation
# ---------------------------------------------------------------------------
def _english_label() -> Label:
    return {
        "source": "sroie",
        "company": "TESCO STORE",
        "total": "9.00",
        "words": [
            {"text": "TESCO STORE", "box": [10, 10, 100, 30]},
            {"text": "MILK 1.50", "box": [10, 60, 100, 80]},
            {"text": "TOTAL 9.00", "box": [10, 200, 120, 220]},
            {"text": "CASH 10.00", "box": [10, 240, 120, 260]},
            {"text": "CHANGE 1.00", "box": [10, 280, 120, 300]},
        ],
    }


def test_generate_german_rewrites_keywords_and_amounts() -> None:
    german = generate_german_from_label(_english_label(), seed=0)
    assert german["source"] == "synthetic"
    texts = " ".join(w["text"] for w in german["words"])

    # English keywords are gone, German ones present.
    assert "TOTAL" not in texts
    assert "CASH" not in texts
    assert "CHANGE" not in texts
    assert "BAR" in texts
    assert "RÜCKGELD" in texts
    assert any(kw in texts for kw in ("SUMME", "ZU ZAHLEN", "GESAMT"))

    # Amounts reformatted to German with euro.
    assert "9,00 €" in texts
    assert "1,50 €" in texts


def test_generate_german_store_and_total_appear_in_words() -> None:
    german = generate_german_from_label(_english_label(), seed=1)
    company = german["company"]
    assert company is not None
    # The store name is German-style and appears as its own word.
    assert any(kw in company for kw in ("GmbH", "KG", "e.K.", "AG"))
    assert any(w["text"] == company for w in german["words"])

    # The labelled total value appears among the words (so it is trainable).
    target = parse_amount(german["total"] or "")
    found = any(
        parse_amount(tok) == target
        for w in german["words"]
        for tok in w["text"].split()
    )
    assert found


def test_render_words_image_is_valid_jpeg() -> None:
    words = [{"text": "HELLO", "box": [10, 10, 100, 30]}]
    data = render_words_image(words)
    img = Image.open(io.BytesIO(data))
    assert img.format == "JPEG"


# ---------------------------------------------------------------------------
# Offline generation against a temporary folder (what CI runs)
# ---------------------------------------------------------------------------
def test_generate_german_receipts_offline(tmp_path: Path) -> None:
    write_example(tmp_path, "sroie-1", b"img", _english_label())

    written = generate_german_receipts(tmp_path)
    assert written == 1
    assert example_exists(tmp_path, "sroie-1-de")

    labels = dict(iter_labels(tmp_path))
    german = labels["sroie-1-de"]
    assert german["source"] == "synthetic"
    assert any(w["text"] == german["company"] for w in german["words"])
    assert (tmp_path / "images" / "sroie-1-de.jpg").exists()


def test_generate_german_receipts_is_idempotent(tmp_path: Path) -> None:
    write_example(tmp_path, "sroie-1", b"img", _english_label())
    assert generate_german_receipts(tmp_path) == 1
    # Re-running does not duplicate and skips the already-generated receipt.
    assert generate_german_receipts(tmp_path) == 0


def test_generate_skips_non_public_sources(tmp_path: Path) -> None:
    label = _english_label()
    label["source"] = "synthetic"
    write_example(tmp_path, "already-de", b"img", label)
    assert generate_german_receipts(tmp_path) == 0
