"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def clear_s3_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "S3_BUCKET",
        "S3_REGION",
        "S3_ENDPOINT_URL",
        "S3_PUBLIC_URL_BASE",
        "S3_KEY_PREFIX",
    ):
        monkeypatch.delenv(key, raising=False)
