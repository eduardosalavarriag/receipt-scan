from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.helpers import (
    fake_heic_bytes,
    make_plain_photo,
    make_receipt_photo,
    make_text_receipt_photo,
)

pytestmark = pytest.mark.usefixtures("clear_s3_env")


def _decode_b64_png(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def test_scan_returns_grayscale_cropped_upright(client: TestClient) -> None:
    photo = make_receipt_photo(angle=14.0, canvas=(900, 1200), receipt=(360, 720))
    resp = client.post(
        "/scan", files={"file": ("receipt.jpg", photo, "image/jpeg")}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "image" in body

    img = _decode_b64_png(body["image"])
    arr = np.asarray(img.convert("L"))

    # Grayscale, not pure black/white: there must be intermediate gray values.
    mid = np.count_nonzero((arr > 40) & (arr < 215))
    assert mid > arr.size * 0.05

    # Cropped: smaller than the original 900x1200 canvas.
    h, w = arr.shape
    assert h < 1200 and w < 900
    # Upright portrait receipt: taller than wide.
    assert h > w


def test_image_false_omits_image(client: TestClient) -> None:
    photo = make_receipt_photo()
    resp = client.post(
        "/scan",
        files={"file": ("receipt.jpg", photo, "image/jpeg")},
        params={"image": "false"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "image" not in body
    assert "store" in body and "total" in body


def test_scan_reads_store_and_total(client: TestClient) -> None:
    photo = make_text_receipt_photo(
        ["ALDI SUED", "Milch 1,19", "Brot 2,49", "SUMME 51,76 EUR"]
    )
    resp = client.post(
        "/scan", files={"file": ("receipt.jpg", photo, "image/jpeg")}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["store"]["value"] == "ALDI SUED"
    assert body["store"]["confidence"] > 0
    assert body["total"]["value"] == "51.76"
    assert body["total"]["currency"] == "EUR"
    assert body["total"]["confidence"] > 0


def test_scan_no_amounts_returns_nulls_low_confidence(client: TestClient) -> None:
    photo = make_plain_photo(size=(640, 480))
    resp = client.post(
        "/scan", files={"file": ("plain.jpg", photo, "image/jpeg")}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"]["value"] is None
    assert body["total"]["currency"] is None
    assert body["total"]["confidence"] == 0.0


def test_ocr_lang_env_changes_tesseract_languages(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, str] = {}

    def fake_ocr(gray: object, lang: str) -> str:
        captured["lang"] = lang
        return ""

    monkeypatch.setattr("app.main.run_ocr", fake_ocr)
    monkeypatch.setenv("OCR_LANG", "fra+eng")
    photo = make_receipt_photo()
    resp = client.post(
        "/scan", files={"file": ("receipt.jpg", photo, "image/jpeg")}
    )
    assert resp.status_code == 200, resp.text
    assert captured["lang"] == "fra+eng"


def test_plain_photo_not_wrongly_cropped(client: TestClient) -> None:
    photo = make_plain_photo(size=(640, 480))
    resp = client.post(
        "/scan", files={"file": ("plain.jpg", photo, "image/jpeg")}
    )
    assert resp.status_code == 200, resp.text
    img = _decode_b64_png(resp.json()["image"])
    # No confident receipt outline -> returned whole, same dimensions.
    assert img.size == (640, 480)


def test_heic_rejected_with_415(client: TestClient) -> None:
    resp = client.post(
        "/scan", files={"file": ("photo.heic", fake_heic_bytes(), "image/heic")}
    )
    assert resp.status_code == 415
    assert "JPEG" in resp.json()["error"]


def test_non_image_rejected_with_400(client: TestClient) -> None:
    resp = client.post(
        "/scan", files={"file": ("notes.txt", b"this is not an image", "text/plain")}
    )
    assert resp.status_code == 400


def test_oversize_rejected_with_413(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1024")
    photo = make_receipt_photo()
    assert len(photo) > 1024
    resp = client.post(
        "/scan", files={"file": ("receipt.jpg", photo, "image/jpeg")}
    )
    assert resp.status_code == 413
