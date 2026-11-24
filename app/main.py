"""FastAPI application exposing /health and /scan."""

from __future__ import annotations

import base64
import hmac
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from . import __version__
from .config import Settings, load_data_dir, load_settings, validate_settings
from .extract import extract_fields
from .imaging import (
    DecodeError,
    UnsupportedImageError,
    decode_image,
    encode_png,
    process_receipt,
)
from .model import load_cached_model
from .ocr import run_ocr, run_ocr_words
from .storage import StorageError, upload_png

# Endpoints reachable without a token even when API_TOKEN is set.
_PUBLIC_PATHS = frozenset({"/health"})


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Refuse to start on invalid configuration (e.g. LABELING without a token).
    validate_settings(load_settings())
    yield


app = FastAPI(title="receipt-scan", version=__version__, lifespan=lifespan)


def get_settings() -> Settings:
    # Read fresh so tests can patch the environment per-request.
    return load_settings()


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


@app.middleware("http")
async def require_api_token(
    request: Request, call_next: Callable[[Request], Awaitable[object]]
) -> object:
    """Guard every endpoint but /health with a bearer token when configured."""
    settings = get_settings()
    if settings.api_token is not None and request.url.path not in _PUBLIC_PATHS:
        header = request.headers.get("authorization", "")
        expected = f"Bearer {settings.api_token}"
        if not hmac.compare_digest(header, expected):
            return _error(401, "A valid API token is required.")
    return await call_next(request)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/scan")
async def scan(
    request: Request,
    file: UploadFile = File(...),
    image: bool = Query(True),
) -> JSONResponse:
    settings = get_settings()

    # Reject oversized uploads up front when the client declares the length.
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > settings.max_upload_bytes:
                return _error(413, "The uploaded image is too large.")
        except ValueError:
            pass

    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        return _error(413, "The uploaded image is too large.")
    if not data:
        return _error(400, "The upload is empty.")

    try:
        rgb = decode_image(data)
    except UnsupportedImageError as exc:
        return _error(415, str(exc))
    except DecodeError as exc:
        return _error(400, str(exc))

    processed = process_receipt(rgb)
    model = load_cached_model(load_data_dir())
    if model is not None:
        # ML pickers: score each OCR word/line; confidence is the picker's score.
        words = run_ocr_words(processed, settings.ocr_lang)
        fields = model.extract_fields(words)
    else:
        # No model: the simple rules over the plain OCR text.
        ocr_text = run_ocr(processed, settings.ocr_lang)
        fields = extract_fields(ocr_text)

    body: dict[str, object] = {
        "store": fields["store"],
        "total": fields["total"],
    }

    if image:
        png = encode_png(processed)
        if settings.s3 is not None:
            try:
                key, url = upload_png(settings.s3, png)
            except StorageError:
                return _error(502, "Failed to store the image.")
            body["key"] = key
            body["image_url"] = url
        else:
            body["image"] = base64.b64encode(png).decode("ascii")

    return JSONResponse(status_code=200, content=body)
