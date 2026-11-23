"""Turn OCR text into a store name and a total, using simple rules.

This is the pre-ML extractor. The store is the top text line; the total is the
amount on a ``SUMME`` / ``ZU ZAHLEN`` / ``GESAMT`` / ``TOTAL`` line, falling back
to the largest amount on the receipt. Amounts are parsed into exact decimals
(never floats) and returned as decimal strings with a dot.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

# Keywords that mark the line carrying the grand total. Checked case-insensitively
# as substrings so "GESAMTBETRAG" and "ZU ZAHLEN" both match.
TOTAL_KEYWORDS = ("ZU ZAHLEN", "SUMME", "GESAMT", "TOTAL")

# Currency symbols/codes we recognise, in priority order. The first match in the
# text wins. Each entry maps a symbol or code to its ISO code.
_CURRENCIES: tuple[tuple[str, str], ...] = (
    ("€", "EUR"),
    ("EUR", "EUR"),
    ("$", "USD"),
    ("USD", "USD"),
    ("£", "GBP"),
    ("GBP", "GBP"),
    ("CHF", "CHF"),
)

# A run of digits that may contain grouping/decimal separators. Matches
# "51,76", "1.234,50", "60.000", "2,49" and bare integers like "5".
_NUMBER_RE = re.compile(r"\d[\d.,]*\d|\d")

# Confidence attached to answers found by these simple rules.
_RULE_CONFIDENCE = 0.5


def parse_amount(token: str) -> Decimal | None:
    """Parse a money token into an exact :class:`~decimal.Decimal`.

    Handles both European (``1.234,50``) and English (``1,234.50``) separators,
    plain decimals (``51,76``) and grouped integers (``60.000`` → ``60000``).
    Returns ``None`` when the token holds no parseable number.
    """
    core = re.sub(r"[^\d.,]", "", token).strip(".,")
    if not core or not any(c.isdigit() for c in core):
        return None

    has_dot = "." in core
    has_comma = "," in core

    if has_dot and has_comma:
        # Both separators present: the rightmost one is the decimal separator.
        if core.rfind(".") > core.rfind(","):
            decimal_sep, group_sep = ".", ","
        else:
            decimal_sep, group_sep = ",", "."
        normalized = core.replace(group_sep, "").replace(decimal_sep, ".")
    elif has_dot or has_comma:
        sep = "." if has_dot else ","
        if core.count(sep) > 1:
            # Several of one separator can only be grouping: "1.234.567".
            normalized = core.replace(sep, "")
        else:
            fraction = core.split(sep)[1]
            if len(fraction) == 3:
                # A single separator with three trailing digits groups thousands.
                normalized = core.replace(sep, "")
            else:
                normalized = core.replace(sep, ".")
    else:
        normalized = core

    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def detect_currency(text: str) -> str | None:
    """Return the ISO currency code for the first symbol/code found, or None."""
    upper = text.upper()
    for symbol, code in _CURRENCIES:
        if symbol.upper() in upper:
            return code
    return None


def _amounts_in(line: str) -> list[Decimal]:
    parsed = (parse_amount(tok) for tok in _NUMBER_RE.findall(line))
    return [amount for amount in parsed if amount is not None]


def _format(amount: Decimal) -> str:
    """Render a decimal as a plain string (e.g. ``Decimal('51.76')`` → ``51.76``)."""
    return format(amount, "f")


def extract_fields(text: str) -> dict[str, object]:
    """Extract store and total fields from OCR text.

    Returns a dict shaped as::

        {"store": {"value": ..., "confidence": ...},
         "total": {"value": ..., "currency": ..., "confidence": ...}}

    ``value`` is ``None`` with low confidence when nothing could be found.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    store = lines[0] if lines else None

    all_amounts: list[Decimal] = []
    keyword_amounts: list[Decimal] = []
    keyword_line: str | None = None
    for line in lines:
        amounts = _amounts_in(line)
        all_amounts.extend(amounts)
        if amounts and any(kw in line.upper() for kw in TOTAL_KEYWORDS):
            keyword_amounts.extend(amounts)
            if keyword_line is None:
                keyword_line = line

    if keyword_amounts:
        total: Decimal | None = max(keyword_amounts)
        currency = detect_currency(keyword_line or "") or detect_currency(text)
    elif all_amounts:
        total = max(all_amounts)
        currency = detect_currency(text)
    else:
        total = None
        currency = None

    store_field: dict[str, object] = {
        "value": store,
        "confidence": _RULE_CONFIDENCE if store else 0.0,
    }
    if total is not None:
        total_field: dict[str, object] = {
            "value": _format(total),
            "currency": currency,
            "confidence": _RULE_CONFIDENCE,
        }
    else:
        total_field = {"value": None, "currency": None, "confidence": 0.0}

    return {"store": store_field, "total": total_field}
