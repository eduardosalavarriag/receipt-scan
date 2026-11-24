from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.dataset import iter_labels
from tests.helpers import make_text_receipt_photo

pytestmark = pytest.mark.usefixtures("clear_s3_env")

TOKEN = "s3cr3t-token"


@pytest.fixture()
def labeling_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Turn labelling on with a token and a throwaway data directory."""
    monkeypatch.setenv("LABELING", "true")
    monkeypatch.setenv("API_TOKEN", TOKEN)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return tmp_path


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def test_label_page_404_when_labeling_off(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LABELING", raising=False)
    resp = client.get("/label")
    assert resp.status_code == 404


def test_label_page_served_when_labeling_on(
    client: TestClient, labeling_env: Path
) -> None:
    # The shell loads without a token; the page asks for it and uses it for API calls.
    resp = client.get("/label")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<form" in resp.text or "type=\"file\"" in resp.text


def test_suggest_returns_cleaned_image_and_fields(
    client: TestClient, labeling_env: Path
) -> None:
    photo = make_text_receipt_photo(
        ["ALDI SUED", "Milch 1,19", "Brot 2,49", "SUMME 51,76 EUR"]
    )
    resp = client.post(
        "/label/suggest",
        files={"file": ("receipt.jpg", photo, "image/jpeg")},
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["image"]  # base64 cleaned image
    base64.b64decode(body["image"])  # decodes
    assert "store" in body and "total" in body
    assert isinstance(body["words"], list) and body["words"]


def test_save_writes_example_train_evaluate_accept(
    client: TestClient, labeling_env: Path
) -> None:
    data_dir = labeling_env
    photo = make_text_receipt_photo(["ALDI SUED", "SUMME 51,76 EUR"])
    suggest = client.post(
        "/label/suggest",
        files={"file": ("receipt.jpg", photo, "image/jpeg")},
        headers=_auth(),
    ).json()

    resp = client.post(
        "/label/save",
        json={
            "image": suggest["image"],
            "store": "ALDI SÜD",
            "total": "51.76",
            "currency": "EUR",
            "words": suggest["words"],
        },
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text

    labels = list(iter_labels(data_dir))
    assert len(labels) == 1
    _id, label = labels[0]
    assert label["source"] == "labelled"
    assert label["company"] == "ALDI SÜD"
    assert label["total"] == "51.76"
    assert label["words"] == suggest["words"]

    # The written image exists on disk.
    images = list((data_dir / "images").glob("*"))
    assert len(images) == 1


def test_save_stores_corrected_values(
    client: TestClient, labeling_env: Path
) -> None:
    data_dir = labeling_env
    photo = make_text_receipt_photo(["WRONG STORE", "SUMME 9,99 EUR"])
    suggest = client.post(
        "/label/suggest",
        files={"file": ("receipt.jpg", photo, "image/jpeg")},
        headers=_auth(),
    ).json()

    resp = client.post(
        "/label/save",
        json={
            "image": suggest["image"],
            "store": "Corrected Store",
            "total": "123.45",
            "currency": "EUR",
            "words": suggest["words"],
        },
        headers=_auth(),
    )
    assert resp.status_code == 200, resp.text

    _id, label = next(iter_labels(data_dir))
    assert label["company"] == "Corrected Store"
    assert label["total"] == "123.45"


def test_label_requests_require_token(
    client: TestClient, labeling_env: Path
) -> None:
    photo = make_text_receipt_photo(["ALDI SUED", "SUMME 51,76 EUR"])
    suggest = client.post(
        "/label/suggest",
        files={"file": ("receipt.jpg", photo, "image/jpeg")},
    )
    assert suggest.status_code == 401

    save = client.post(
        "/label/save",
        json={"image": "", "store": "x", "total": "1", "words": []},
    )
    assert save.status_code == 401
