from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import ConfigError, load_settings, validate_settings
from app.main import app
from tests.helpers import make_receipt_photo

pytestmark = pytest.mark.usefixtures("clear_s3_env")

TOKEN = "s3cr3t-token"


@pytest.fixture()
def stub_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep /scan off the real OCR engine so auth tests need no Tesseract."""
    monkeypatch.setattr("app.main.run_ocr", lambda gray, lang: "")
    monkeypatch.setattr("app.main.load_cached_model", lambda data_dir: None)


def test_endpoints_open_without_api_token(
    client: TestClient, stub_ocr: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("API_TOKEN", raising=False)
    resp = client.post(
        "/scan",
        files={"file": ("receipt.jpg", make_receipt_photo(), "image/jpeg")},
        params={"image": "false"},
    )
    assert resp.status_code == 200, resp.text


def test_missing_token_rejected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("API_TOKEN", TOKEN)
    resp = client.post(
        "/scan",
        files={"file": ("receipt.jpg", make_receipt_photo(), "image/jpeg")},
        params={"image": "false"},
    )
    assert resp.status_code == 401
    assert "token" in resp.json()["error"].lower()


def test_wrong_token_rejected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("API_TOKEN", TOKEN)
    resp = client.post(
        "/scan",
        files={"file": ("receipt.jpg", make_receipt_photo(), "image/jpeg")},
        params={"image": "false"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401


def test_correct_token_allows_request(
    client: TestClient, stub_ocr: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("API_TOKEN", TOKEN)
    resp = client.post(
        "/scan",
        files={"file": ("receipt.jpg", make_receipt_photo(), "image/jpeg")},
        params={"image": "false"},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert resp.status_code == 200, resp.text


def test_health_always_open(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("API_TOKEN", TOKEN)
    resp = client.get("/health")
    assert resp.status_code == 200


def test_validate_settings_labeling_requires_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LABELING", "true")
    monkeypatch.delenv("API_TOKEN", raising=False)
    with pytest.raises(ConfigError):
        validate_settings(load_settings())


def test_validate_settings_labeling_with_token_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LABELING", "true")
    monkeypatch.setenv("API_TOKEN", TOKEN)
    validate_settings(load_settings())  # no exception


def test_startup_fails_when_labeling_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LABELING", "true")
    monkeypatch.delenv("API_TOKEN", raising=False)
    with pytest.raises(ConfigError):
        with TestClient(app):
            pass
