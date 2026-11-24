from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.dataset import Label, Word, write_example
from app.extract import extract_fields
from app.evaluate import reconstruct_text
from app.model import find_model_path, load_cached_model, load_model, model_file
from app.train import train_model
from tests.helpers import make_receipt_photo

# A distinctive store token that must NOT survive into the saved model: it proves
# the text features store no vocabulary (they are hashed, not kept).
UNIQUE_STORE_TOKEN = "ZZUNIQBRANDXQ"

# The total line uses a keyword the simple rules do NOT know ("BETRAG"), and the
# receipts carry a larger decoy amount with no keyword. The rules fall back to the
# largest amount and get it wrong; the model must learn to pick the BETRAG line.
TOTAL_KEYWORD = "BETRAG"


def _receipt(index: int, *, total: str, decoy: str) -> Label:
    store = f"{UNIQUE_STORE_TOKEN} MARKT {index}"
    total_de = total.replace(".", ",")
    decoy_de = decoy.replace(".", ",")
    words: list[Word] = [
        {"text": store, "box": [10, 10, 260, 40]},
        {"text": f"RABATT {decoy_de} EUR", "box": [10, 120, 260, 150]},
        {"text": f"{TOTAL_KEYWORD} {total_de} EUR", "box": [10, 300, 260, 330]},
    ]
    return {
        "source": "synthetic",
        "company": store,
        "total": total,
        "words": words,
    }


def _make_dataset(tmp_path: Path, n: int = 30) -> None:
    for i in range(n):
        total = f"{i + 1}.{(i * 7) % 100:02d}"
        decoy = f"{i + 80}.{(i * 3) % 100:02d}"
        write_example(tmp_path, f"r{i}", b"img", _receipt(i, total=total, decoy=decoy))


def _fresh_words() -> list[Word]:
    """A receipt the model has never seen: BETRAG 7,00 is the real total, 99,00 a decoy."""
    return [
        {"text": f"{UNIQUE_STORE_TOKEN} MARKT NEU", "box": [10, 10, 260, 40]},
        {"text": "RABATT 99,00 EUR", "box": [10, 120, 260, 150]},
        {"text": f"{TOTAL_KEYWORD} 7,00 EUR", "box": [10, 300, 260, 330]},
    ]


# ---------------------------------------------------------------------------
# train writes a model and reports a held-out score
# ---------------------------------------------------------------------------
def test_train_writes_model_and_scores_holdout(tmp_path: Path) -> None:
    _make_dataset(tmp_path)

    result = train_model(tmp_path)

    assert model_file(tmp_path).exists()
    assert result.holdout_receipts >= 1
    assert result.train_receipts >= 1
    assert result.store_rows > 0 and result.total_rows > 0
    # A held-out score was computed over receipts not used for training.
    assert result.evaluation.groups == result.holdout_receipts


def test_train_on_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        train_model(tmp_path)


# ---------------------------------------------------------------------------
# the model beats the simple rules on a larger non-total amount
# ---------------------------------------------------------------------------
def test_model_picks_total_where_rules_fail(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    train_model(tmp_path)
    model = load_model(model_file(tmp_path))

    words = _fresh_words()

    # The simple rules fall back to the largest amount (the 99,00 decoy).
    rules_total = extract_fields(reconstruct_text(words))["total"]["value"]
    assert rules_total == "99.00"

    # The model learns the real total line and picks 7,00, with a real confidence.
    value, currency, confidence = model.predict_total(words)
    assert value == "7.00"
    assert currency == "EUR"
    assert confidence > 0.0


# ---------------------------------------------------------------------------
# the saved model stores no word vocabulary
# ---------------------------------------------------------------------------
def test_saved_model_has_no_word_vocabulary(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    train_model(tmp_path)

    raw = model_file(tmp_path).read_bytes()
    assert UNIQUE_STORE_TOKEN.encode() not in raw

    model = load_model(model_file(tmp_path))
    assert getattr(model.vectorizer, "vocabulary_", None) is None


# ---------------------------------------------------------------------------
# model discovery / caching
# ---------------------------------------------------------------------------
def test_no_model_file_means_no_model(tmp_path: Path) -> None:
    assert find_model_path(tmp_path) is None
    assert load_cached_model(tmp_path) is None


# ---------------------------------------------------------------------------
# the service uses the model when present
# ---------------------------------------------------------------------------
def test_scan_uses_model_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_dataset(tmp_path)
    train_model(tmp_path)

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for key in ("S3_BUCKET", "S3_REGION", "S3_ENDPOINT_URL", "S3_PUBLIC_URL_BASE"):
        monkeypatch.delenv(key, raising=False)
    # Avoid Tesseract: feed the service the OCR words directly.
    monkeypatch.setattr("app.main.run_ocr_words", lambda gray, lang: _fresh_words())

    from app.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/scan", files={"file": ("receipt.jpg", make_receipt_photo(), "image/jpeg")}
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"]["value"] == "7.00"
    assert body["total"]["confidence"] > 0.0
    assert UNIQUE_STORE_TOKEN in body["store"]["value"]
    # Image still delivered inline (no S3 configured).
    assert Image.open(io.BytesIO(base64.b64decode(body["image"])))
