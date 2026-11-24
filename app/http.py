"""Shared HTTP helpers for the image-upload endpoints.

The read-size-decode step is identical for ``/scan`` and ``/label/suggest``, so
it lives here once. :func:`read_upload_image` returns the decoded RGB array or
raises :class:`UploadError` carrying the status and message each endpoint turns
into its own error response.
"""

from __future__ import annotations

import numpy as np
from fastapi import UploadFile

from .config import Settings
from .imaging import DecodeError, UnsupportedImageError, decode_image


class UploadError(Exception):
    """An uploaded image was rejected; carries the HTTP status and message."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


async def read_upload_image(file: UploadFile, settings: Settings) -> np.ndarray:
    """Read and decode an uploaded image into an RGB array.

    Raises :class:`UploadError` when the upload is too large (413), empty (400),
    an unsupported image type such as HEIC (415), or not a decodable image (400).
    """
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise UploadError(413, "The uploaded image is too large.")
    if not data:
        raise UploadError(400, "The upload is empty.")
    try:
        return decode_image(data)
    except UnsupportedImageError as exc:
        raise UploadError(415, str(exc)) from exc
    except DecodeError as exc:
        raise UploadError(400, str(exc)) from exc
