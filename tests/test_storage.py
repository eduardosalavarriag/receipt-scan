from __future__ import annotations

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from tests.helpers import make_receipt_photo


@pytest.fixture()
def s3_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S3_BUCKET", "receipts")
    monkeypatch.setenv("S3_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("S3_PUBLIC_URL_BASE", raising=False)


def test_scan_uploads_to_s3(client: TestClient, s3_env: None) -> None:
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="receipts")
        resp = client.post(
            "/scan", files={"file": ("receipt.jpg", make_receipt_photo(), "image/jpeg")}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "image" not in body
        assert body["key"].startswith("scans/")
        assert body["key"].endswith(".png")
        assert body["image_url"].endswith(body["key"])

        stored = boto3.client("s3", region_name="us-east-1").get_object(
            Bucket="receipts", Key=body["key"]
        )
        assert stored["Body"].read()[:4] == b"\x89PNG"


def test_storage_failure_returns_502(client: TestClient, s3_env: None) -> None:
    # No bucket created under moto -> put_object fails -> 502.
    with mock_aws():
        resp = client.post(
            "/scan", files={"file": ("receipt.jpg", make_receipt_photo(), "image/jpeg")}
        )
        assert resp.status_code == 502


def test_s3_image_false_skips_upload(client: TestClient, s3_env: None) -> None:
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="receipts")
        resp = client.post(
            "/scan",
            files={"file": ("receipt.jpg", make_receipt_photo(), "image/jpeg")},
            params={"image": "false"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "image" not in body
        assert "image_url" not in body and "key" not in body
        assert "store" in body and "total" in body
