"""The ``/label`` page: collect corrected labels into the dataset.

Enabled only when ``LABELING=true`` (which also requires ``API_TOKEN`` — see
:func:`app.config.validate_settings`). One plain, phone-friendly HTML page lets
you take or upload a photo, see the cleaned image with the store and total the
ML pickers suggest, correct them, and save. Each save writes an image plus a
label (with the OCR words and boxes) into ``/data`` in the dataset format with
``source: "labelled"``, so ``train`` and ``evaluate`` pick it up like any other
example.

The browser never sends the image back for OCR a second time: ``/label/suggest``
returns the cleaned image and the OCR words, and ``/label/save`` stores exactly
those words alongside the (possibly corrected) store and total.
"""

from __future__ import annotations

import base64
import uuid
from typing import Any

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from .config import load_data_dir, load_settings
from .dataset import Label, Word, write_example
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

router = APIRouter()


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


def _not_found() -> JSONResponse:
    return _error(404, "Not found.")


def _clean(value: Any) -> str | None:
    """A trimmed string, or ``None`` when empty/missing (how blanks are stored)."""
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed or None


def _coerce_words(raw: Any) -> list[Word]:
    """Validate the words round-tripped from ``/label/suggest`` back to us."""
    if not isinstance(raw, list):
        raise ValueError("words must be a list")
    words: list[Word] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each word must be an object")
        text = item.get("text")
        box = item.get("box")
        if not isinstance(text, str) or not isinstance(box, list) or len(box) != 4:
            raise ValueError("each word needs text and a 4-number box")
        words.append({"text": text, "box": [int(round(float(c))) for c in box]})
    return words


@router.get("/label", response_model=None)
def label_page() -> HTMLResponse | JSONResponse:
    """The labelling page shell. 404 unless ``LABELING=true``.

    Served without a token so the browser can load it; the page then asks for the
    token once and sends it on every ``/label/suggest`` and ``/label/save`` call,
    both of which the API-token middleware guards.
    """
    if not load_settings().labeling:
        return _not_found()
    return HTMLResponse(content=LABEL_PAGE_HTML)


@router.post("/label/suggest")
async def label_suggest(file: UploadFile = File(...)) -> JSONResponse:
    """Clean a photo and suggest a store and total, returning the OCR words too."""
    settings = load_settings()
    if not settings.labeling:
        return _not_found()

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
    words = run_ocr_words(processed, settings.ocr_lang)

    model = load_cached_model(load_data_dir())
    if model is not None:
        fields = model.extract_fields(words)
    else:
        fields = extract_fields(run_ocr(processed, settings.ocr_lang))

    return JSONResponse(
        status_code=200,
        content={
            "image": base64.b64encode(encode_png(processed)).decode("ascii"),
            "store": fields["store"]["value"],
            "total": fields["total"]["value"],
            "currency": fields["total"]["currency"],
            "words": words,
        },
    )


@router.post("/label/save")
async def label_save(request: Request) -> JSONResponse:
    """Write a corrected label (image + words + store/total) into ``/data``."""
    settings = load_settings()
    if not settings.labeling:
        return _not_found()

    try:
        payload = await request.json()
    except Exception:
        return _error(400, "The request body must be JSON.")
    if not isinstance(payload, dict):
        return _error(400, "The request body must be a JSON object.")

    image_b64 = payload.get("image")
    if not isinstance(image_b64, str) or not image_b64:
        return _error(400, "An image is required.")
    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
    except Exception:
        return _error(400, "The image is not valid base64.")
    if not image_bytes:
        return _error(400, "The image is empty.")

    try:
        words = _coerce_words(payload.get("words", []))
    except ValueError as exc:
        return _error(400, str(exc))

    label: Label = {
        "source": "labelled",
        "company": _clean(payload.get("store")),
        "total": _clean(payload.get("total")),
        "words": words,
    }

    example_id = f"labelled-{uuid.uuid4().hex}"
    write_example(
        load_data_dir(), example_id, image_bytes, label, image_ext="png"
    )
    return JSONResponse(status_code=200, content={"id": example_id, "saved": True})


# The page is intentionally one plain, dependency-free HTML file: a file input
# (camera on phones), the cleaned image, editable store/total fields, and Save.
# The token is asked for once and kept in localStorage.
LABEL_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label a receipt</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; padding: 1rem;
         max-width: 40rem; margin: 0 auto; background: #fafafa; color: #111; }
  h1 { font-size: 1.3rem; }
  label { display: block; margin: 0.75rem 0 0.25rem; font-weight: 600; }
  input[type=text], input[type=file] { width: 100%; padding: 0.6rem;
         font-size: 1rem; box-sizing: border-box; border: 1px solid #bbb;
         border-radius: 0.4rem; }
  button { margin-top: 1rem; width: 100%; padding: 0.8rem; font-size: 1rem;
         border: 0; border-radius: 0.4rem; background: #2563eb; color: #fff; }
  button:disabled { background: #9ab; }
  img#preview { max-width: 100%; margin-top: 1rem; border: 1px solid #ccc;
         border-radius: 0.4rem; display: none; }
  #status { margin-top: 1rem; min-height: 1.2rem; }
  .error { color: #b91c1c; }
  .ok { color: #15803d; }
</style>
</head>
<body>
<h1>Label a receipt</h1>
<form id="form">
  <label for="photo">Receipt photo</label>
  <input type="file" id="photo" accept="image/*" capture="environment">
  <img id="preview" alt="Cleaned receipt">
  <label for="store">Store</label>
  <input type="text" id="store" autocomplete="off">
  <label for="total">Total</label>
  <input type="text" id="total" inputmode="decimal" autocomplete="off">
  <button type="submit" id="save" disabled>Save</button>
</form>
<p id="status"></p>
<script>
(function () {
  var statusEl = document.getElementById("status");
  var preview = document.getElementById("preview");
  var storeEl = document.getElementById("store");
  var totalEl = document.getElementById("total");
  var saveBtn = document.getElementById("save");
  var photo = document.getElementById("photo");
  var state = { image: null, words: [], currency: null };

  function token() {
    var t = localStorage.getItem("receiptScanToken");
    while (!t) {
      t = window.prompt("Enter the API token");
      if (t) { localStorage.setItem("receiptScanToken", t); }
      else { return null; }
    }
    return t;
  }

  function authHeaders(extra) {
    var h = extra || {};
    h["Authorization"] = "Bearer " + (token() || "");
    return h;
  }

  function setStatus(msg, cls) {
    statusEl.textContent = msg;
    statusEl.className = cls || "";
  }

  photo.addEventListener("change", function () {
    var file = photo.files[0];
    if (!file) { return; }
    setStatus("Reading receipt...");
    saveBtn.disabled = true;
    var fd = new FormData();
    fd.append("file", file);
    fetch("/label/suggest", { method: "POST", headers: authHeaders(), body: fd })
      .then(function (r) {
        if (r.status === 401) {
          localStorage.removeItem("receiptScanToken");
          throw new Error("Invalid token. Reload and try again.");
        }
        if (!r.ok) { throw new Error("Could not read the receipt."); }
        return r.json();
      })
      .then(function (d) {
        state.image = d.image;
        state.words = d.words || [];
        state.currency = d.currency || null;
        preview.src = "data:image/png;base64," + d.image;
        preview.style.display = "block";
        storeEl.value = d.store || "";
        totalEl.value = d.total || "";
        saveBtn.disabled = false;
        setStatus("Check the suggested store and total, then Save.");
      })
      .catch(function (e) { setStatus(e.message, "error"); });
  });

  document.getElementById("form").addEventListener("submit", function (ev) {
    ev.preventDefault();
    if (!state.image) { return; }
    saveBtn.disabled = true;
    setStatus("Saving...");
    fetch("/label/save", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        image: state.image,
        store: storeEl.value,
        total: totalEl.value,
        currency: state.currency,
        words: state.words
      })
    })
      .then(function (r) {
        if (r.status === 401) {
          localStorage.removeItem("receiptScanToken");
          throw new Error("Invalid token. Reload and try again.");
        }
        if (!r.ok) { throw new Error("Could not save."); }
        return r.json();
      })
      .then(function () {
        setStatus("Saved. Take the next photo.", "ok");
        state.image = null;
        preview.style.display = "none";
        photo.value = "";
        storeEl.value = "";
        totalEl.value = "";
      })
      .catch(function (e) { setStatus(e.message, "error"); saveBtn.disabled = false; });
  });
})();
</script>
</body>
</html>"""
