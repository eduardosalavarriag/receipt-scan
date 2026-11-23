"""FastAPI application exposing /health and /scan."""

from __future__ import annotations

import base64

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from . import __version__
from .config import Settings, load_settings
from .extract import extract_fields
from .imaging import (
    DecodeError,
    UnsupportedImageError,
    decode_image,
    encode_png,
    process_receipt,
)
from .ocr import run_ocr
from .storage import StorageError, upload_png

app = FastAPI(title="receipt-scan", version=__version__)


def get_settings() -> Settings:
    # Read fresh so tests can patch the environment per-request.
    return load_settings()


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


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
