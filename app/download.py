"""The ``download`` command: fill ``/data`` with training data.

Three sources, all written in the one dataset format (see :mod:`app.dataset`):

* **SROIE** (ICDAR 2019) and **CORD** (NAVER Clova) — public receipt datasets,
  fetched over the network and converted.
* **Generated German receipts** — word-level rewrites of the public English
  labels (``TOTAL`` → ``SUMME``/``ZU ZAHLEN``/``GESAMT``, ``CASH`` → ``BAR``,
  ``CHANGE`` → ``RÜCKGELD``, ``TAX`` → ``MwSt``, amounts reformatted to
  ``1.234,50`` with ``€``, German-style store names). These run **offline** from
  whatever English labels are already on disk, so they are testable without the
  network and are what CI exercises.

Network fetching is deliberately separated from conversion so the conversion and
generation logic can be unit-tested against a temporary folder; the fetchers are
not run in CI.
"""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
import urllib.request
from decimal import Decimal
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .dataset import Label, Word, iter_labels, write_example
from .extract import parse_amount

# ---------------------------------------------------------------------------
# Tarball sources for the public datasets. These are only used by the network
# fetchers (not run in CI); update them if the mirrors move.
# ---------------------------------------------------------------------------
SROIE_TARBALL_URL = (
    "https://codeload.github.com/zzzDavid/ICDAR-2019-SROIE/tar.gz/refs/heads/master"
)
CORD_TARBALL_URL = (
    "https://codeload.github.com/katanaml/cord/tar.gz/refs/heads/main"
)


# ---------------------------------------------------------------------------
# Amount formatting
# ---------------------------------------------------------------------------
def format_de_amount(amount: Decimal) -> str:
    """Render an amount in German money format: ``1234.5`` → ``1.234,50``."""
    quantised = amount.quantize(Decimal("0.01"))
    sign = "-" if quantised < 0 else ""
    digits = f"{abs(quantised):.2f}"
    int_part, frac = digits.split(".")
    grouped = _group_thousands(int_part)
    return f"{sign}{grouped},{frac}"


def format_de_integer(amount: Decimal) -> str:
    """Render an integer amount with German thousands dots: ``60000`` → ``60.000``."""
    sign = "-" if amount < 0 else ""
    return f"{sign}{_group_thousands(str(abs(int(amount))))}"


def _group_thousands(digits: str) -> str:
    out = []
    for i, ch in enumerate(reversed(digits)):
        if i and i % 3 == 0:
            out.append(".")
        out.append(ch)
    return "".join(reversed(out))


def _is_monetary(token: str) -> bool:
    """A money token has a decimal separator with exactly two trailing digits."""
    core = "".join(c for c in token if c.isdigit() or c in ".,")
    for sep in (".", ","):
        if sep in core:
            frac = core.rsplit(sep, 1)[1]
            if len(frac) == 2 and frac.isdigit():
                return True
    return False


# ---------------------------------------------------------------------------
# German word-level rewriting
# ---------------------------------------------------------------------------
# Whole-word English → German replacements. "TOTAL" is handled separately so it
# can vary between the German synonyms.
_DE_WORD_MAP = {
    "CASH": "BAR",
    "CHANGE": "RÜCKGELD",
    "TAX": "MwSt",
    "VAT": "MwSt",
    "SUBTOTAL": "ZWISCHENSUMME",
    "DATE": "DATUM",
    "TIME": "UHRZEIT",
    "RECEIPT": "BELEG",
    "THANK": "DANKE",
    "YOU": "",
}
_TOTAL_SYNONYMS = ("SUMME", "ZU ZAHLEN", "GESAMT")
_STORE_SUFFIXES = ("GmbH", "KG", "e.K.", "GmbH & Co. KG", "AG")


def german_store_name(company: str | None, seed: int) -> str:
    """Turn an English store name into a deterministic German-style one."""
    base = (company or "Markt").strip().title() or "Markt"
    suffix = _STORE_SUFFIXES[seed % len(_STORE_SUFFIXES)]
    return f"{base} {suffix}"


def _rewrite_token(token: str, seed: int) -> str:
    up = token.upper()
    if up == "TOTAL":
        return _TOTAL_SYNONYMS[seed % len(_TOTAL_SYNONYMS)]
    if up in _DE_WORD_MAP:
        return _DE_WORD_MAP[up]
    amount = parse_amount(token)
    if amount is not None:
        if _is_monetary(token):
            return f"{format_de_amount(amount)} €"
        return format_de_integer(amount)
    return token


def _rewrite_word_text(text: str, seed: int) -> str:
    rewritten = [_rewrite_token(tok, seed) for tok in text.split()]
    return " ".join(tok for tok in rewritten if tok)


def generate_german_from_label(label: Label, seed: int) -> Label:
    """Build a German synthetic label from an English one.

    Every word is rewritten; the store name becomes a German-style name that
    appears as the top word, and the total is guaranteed to appear among the
    words so the result is trainable.
    """
    words: list[Word] = []
    for word in label["words"]:
        new_text = _rewrite_word_text(word["text"], seed)
        if new_text:
            words.append({"text": new_text, "box": list(word["box"])})

    company = german_store_name(label.get("company"), seed)
    words = _ensure_store_word(words, company)

    total = label.get("total")
    words = _ensure_total_word(words, total, seed)

    return {
        "source": "synthetic",
        "company": company,
        "total": total,
        "words": words,
    }


def _ensure_store_word(words: list[Word], company: str) -> list[Word]:
    """Make the top-most word the German store name so the store is trainable."""
    if not words:
        return [{"text": company, "box": [10, 10, 10 + 20 * len(company), 50]}]
    top_index = min(range(len(words)), key=lambda i: words[i]["box"][1])
    words[top_index] = {"text": company, "box": words[top_index]["box"]}
    return words


def _ensure_total_word(words: list[Word], total: str | None, seed: int) -> list[Word]:
    """Guarantee a word carrying the total amount (append one if missing)."""
    if total is None:
        return words
    target = parse_amount(total)
    for word in words:
        if any(parse_amount(tok) == target for tok in word["text"].split()):
            return words
    bottom = max((w["box"][3] for w in words), default=50)
    keyword = _TOTAL_SYNONYMS[seed % len(_TOTAL_SYNONYMS)]
    text = f"{keyword} {format_de_amount(target or Decimal(0))} €"
    words.append({"text": text, "box": [10, bottom + 10, 10 + 18 * len(text), bottom + 50]})
    return words


# ---------------------------------------------------------------------------
# Image rendering for synthetic receipts
# ---------------------------------------------------------------------------
def render_words_image(words: list[Word]) -> bytes:
    """Render words at their box positions onto a white receipt, as JPEG bytes."""
    if words:
        width = max(w["box"][2] for w in words) + 20
        height = max(w["box"][3] for w in words) + 20
    else:
        width, height = 200, 200
    img = Image.new("RGB", (max(width, 40), max(height, 40)), (250, 250, 248))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for word in words:
        draw.text((word["box"][0], word["box"][1]), word["text"], fill=(15, 15, 15), font=font)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Offline generation (what CI tests)
# ---------------------------------------------------------------------------
def generate_german_receipts(data_dir: Path | str) -> int:
    """Generate a German receipt for each English SROIE/CORD label on disk.

    Runs entirely offline from existing labels. Idempotent: the generated id is
    ``<source-id>-de`` so re-running does not duplicate. Returns the count
    written this run.
    """
    written = 0
    for index, (example_id, label) in enumerate(list(iter_labels(data_dir))):
        if label.get("source") not in ("sroie", "cord"):
            continue
        german = generate_german_from_label(label, seed=index)
        image = render_words_image(german["words"])
        if write_example(data_dir, f"{example_id}-de", image, german):
            written += 1
    return written


# ---------------------------------------------------------------------------
# SROIE / CORD conversion (pure; fed by the fetchers)
# ---------------------------------------------------------------------------
def _quad_to_box(coords: list[int]) -> list[int]:
    xs = coords[0::2]
    ys = coords[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def parse_sroie_box_file(text: str) -> list[Word]:
    """Parse a SROIE OCR file (``x1,y1,...,x4,y4,transcription`` per line)."""
    words: list[Word] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", 8)
        if len(parts) < 9:
            continue
        try:
            coords = [int(float(p)) for p in parts[:8]]
        except ValueError:
            continue
        transcription = parts[8].strip()
        if transcription:
            words.append({"text": transcription, "box": _quad_to_box(coords)})
    return words


def convert_sroie(box_text: str, key_json: dict) -> Label:
    total_raw = key_json.get("total")
    total = _normalise_total(total_raw)
    return {
        "source": "sroie",
        "company": (key_json.get("company") or None),
        "total": total,
        "words": parse_sroie_box_file(box_text),
    }


def convert_cord(annotation_json: dict) -> Label:
    """Convert one CORD annotation (``valid_line`` list) to a label."""
    words: list[Word] = []
    total_words: list[str] = []
    for line in annotation_json.get("valid_line", []):
        category = line.get("category", "")
        for item in line.get("words", []):
            quad = item.get("quad", {})
            text = (item.get("text") or "").strip()
            if not text:
                continue
            coords = [
                int(quad.get("x1", 0)), int(quad.get("y1", 0)),
                int(quad.get("x2", 0)), int(quad.get("y2", 0)),
                int(quad.get("x3", 0)), int(quad.get("y3", 0)),
                int(quad.get("x4", 0)), int(quad.get("y4", 0)),
            ]
            words.append({"text": text, "box": _quad_to_box(coords)})
            if category == "total.total_price":
                total_words.append(text)
    total = _normalise_total(" ".join(total_words)) if total_words else None
    company = words[0]["text"] if words else None
    return {"source": "cord", "company": company, "total": total, "words": words}


def _normalise_total(raw: object) -> str | None:
    if raw is None:
        return None
    amount = parse_amount(str(raw))
    if amount is None:
        return None
    return format(amount, "f")


# ---------------------------------------------------------------------------
# Network fetchers (not run in CI)
# ---------------------------------------------------------------------------
def _fetch_tarball(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as resp:  # noqa: S310 - trusted dataset mirror
        data = resp.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")


def download_sroie(data_dir: Path | str) -> int:
    """Fetch SROIE, convert, and write it into the dataset format."""
    written = 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _fetch_tarball(SROIE_TARBALL_URL, root)
        imgs = {p.stem: p for p in root.rglob("*.jpg")}
        boxes = {p.stem: p for p in root.rglob("*.txt") if "/box" in str(p) or "/task1" in str(p)}
        keys = {p.stem: p for p in root.rglob("*.json") if "/key" in str(p) or "/task3" in str(p)}
        for stem, img_path in imgs.items():
            if stem not in boxes or stem not in keys:
                continue
            try:
                key_json = json.loads(keys[stem].read_text(encoding="utf-8", errors="ignore"))
            except json.JSONDecodeError:
                continue
            label = convert_sroie(
                boxes[stem].read_text(encoding="utf-8", errors="ignore"), key_json
            )
            if write_example(data_dir, f"sroie-{stem}", img_path.read_bytes(), label):
                written += 1
    return written


def download_cord(data_dir: Path | str) -> int:
    """Fetch CORD, convert, and write it into the dataset format."""
    written = 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _fetch_tarball(CORD_TARBALL_URL, root)
        imgs = {p.stem: p for p in root.rglob("*.png")}
        imgs.update({p.stem: p for p in root.rglob("*.jpg")})
        for json_path in root.rglob("*.json"):
            stem = json_path.stem
            if stem not in imgs:
                continue
            try:
                annotation = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
            except json.JSONDecodeError:
                continue
            if "valid_line" not in annotation:
                continue
            label = convert_cord(annotation)
            if write_example(data_dir, f"cord-{stem}", imgs[stem].read_bytes(), label):
                written += 1
    return written


def run_download(data_dir: Path | str) -> dict[str, int]:
    """Fetch the public datasets, then generate German receipts from them."""
    counts = {
        "sroie": download_sroie(data_dir),
        "cord": download_cord(data_dir),
    }
    counts["synthetic"] = generate_german_receipts(data_dir)
    return counts
