"""Optional S3-compatible object storage for processed images."""

from __future__ import annotations

import uuid

import boto3
from botocore.config import Config as BotoConfig

from .config import S3Config


class StorageError(Exception):
    """Uploading the processed image to object storage failed."""


def build_key(cfg: S3Config, extension: str = "png") -> str:
    return f"{cfg.key_prefix}/{uuid.uuid4().hex}.{extension}"


def upload_png(cfg: S3Config, data: bytes) -> tuple[str, str]:
    """Upload PNG bytes to S3-compatible storage.

    Returns (key, public_url). Raises StorageError on any failure so the API can
    answer with 502.
    """
    key = build_key(cfg, "png")
    try:
        client = boto3.client(
            "s3",
            region_name=cfg.region,
            endpoint_url=cfg.endpoint_url,
            config=BotoConfig(retries={"max_attempts": 2, "mode": "standard"}),
        )
        client.put_object(
            Bucket=cfg.bucket,
            Key=key,
            Body=data,
            ContentType="image/png",
        )
    except Exception as exc:  # noqa: BLE001 - surface any storage failure as 502
        raise StorageError(str(exc)) from exc
    return key, cfg.public_url(key)
