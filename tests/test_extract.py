from __future__ import annotations

from decimal import Decimal

import pytest

from app.extract import detect_currency, extract_fields, parse_amount


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("1.234,50", Decimal("1234.50")),
        ("1,234.50", Decimal("1234.50")),
        ("51,76", Decimal("51.76")),
        ("60.000", Decimal("60000")),
        ("2,49", Decimal("2.49")),
        ("1.234.567", Decimal("1234567")),
        ("€ 51,76", Decimal("51.76")),
        ("51,76 €", Decimal("51.76")),
        ("5", Decimal("5")),
    ],
)
def test_parse_amount(token: str, expected: Decimal) -> None:
    assert parse_amount(token) == expected


def test_parse_amount_no_float_rounding() -> None:
    # 0.1 + 0.2 style sums are exact with Decimal.
    assert parse_amount("0,10") + parse_amount("0,20") == Decimal("0.30")  # type: ignore[operator]


@pytest.mark.parametrize("token", ["", "abc", "€", "."])
def test_parse_amount_rejects_non_numbers(token: str) -> None:
    assert parse_amount(token) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SUMME 51,76 €", "EUR"),
        ("TOTAL 10 EUR", "EUR"),
        ("TOTAL $10.00", "USD"),
        ("Betrag 5 CHF", "CHF"),
        ("no currency here", None),
    ],
)
def test_detect_currency(text: str, expected: str | None) -> None:
    assert detect_currency(text) == expected


def test_extract_uses_keyword_line_for_total() -> None:
    text = "ALDI SUED\nMilch 1,19\nBrot 2,49\nSUMME 51,76 EUR\n"
    fields = extract_fields(text)
    assert fields["store"] == {"value": "ALDI SUED", "confidence": 0.5}
    assert fields["total"] == {
        "value": "51.76",
        "currency": "EUR",
        "confidence": 0.5,
    }


def test_extract_falls_back_to_largest_amount() -> None:
    text = "Kiosk\nWasser 1,50\nSnack 3,20\n"
    fields = extract_fields(text)
    total = fields["total"]
    assert isinstance(total, dict)
    assert total["value"] == "3.20"


def test_extract_no_amounts_gives_nulls() -> None:
    fields = extract_fields("Just a note\nwith no money\n")
    assert fields["total"] == {"value": None, "currency": None, "confidence": 0.0}


def test_extract_empty_text() -> None:
    fields = extract_fields("")
    assert fields["store"] == {"value": None, "confidence": 0.0}
    assert fields["total"] == {"value": None, "currency": None, "confidence": 0.0}
