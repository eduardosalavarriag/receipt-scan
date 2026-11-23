"""Runtime configuration, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Default maximum upload size: 15 MB. Phone photos are large, but this keeps a
# single request from exhausting memory.
DEFAULT_MAX_UPLOAD_BYTES = 15 * 1024 * 1024


@dataclass(frozen=True)
class S3Config:
    bucket: str
    region: str
    endpoint_url: str | None
    public_url_base: str | None
    key_prefix: str

    def public_url(self, key: str) -> str:
        if self.public_url_base:
            return f"{self.public_url_base.rstrip('/')}/{key}"
        if self.endpoint_url:
            return f"{self.endpoint_url.rstrip('/')}/{self.bucket}/{key}"
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"


@dataclass(frozen=True)
class Settings:
    max_upload_bytes: int
    s3: S3Config | None

    @property
    def storage_enabled(self) -> bool:
        return self.s3 is not None


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_s3_config() -> S3Config | None:
    """Build an S3 config when a bucket is configured, else None."""
    bucket = os.environ.get("S3_BUCKET", "").strip()
    if not bucket:
        return None
    return S3Config(
        bucket=bucket,
        region=os.environ.get("S3_REGION", "us-east-1").strip() or "us-east-1",
        endpoint_url=(os.environ.get("S3_ENDPOINT_URL", "").strip() or None),
        public_url_base=(os.environ.get("S3_PUBLIC_URL_BASE", "").strip() or None),
        key_prefix=os.environ.get("S3_KEY_PREFIX", "scans").strip().strip("/") or "scans",
    )


def load_settings() -> Settings:
    return Settings(
        max_upload_bytes=_get_int("MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES),
        s3=load_s3_config(),
    )
