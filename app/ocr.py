"""Run Tesseract OCR over a processed receipt image."""

from __future__ import annotations

import numpy as np
import pytesseract
from PIL import Image

from .dataset import Word


def run_ocr(gray: np.ndarray, lang: str) -> str:
    """Return the text Tesseract reads from a grayscale image.

    ``lang`` is a Tesseract language spec such as ``deu+eng`` (from ``OCR_LANG``).
    """
    return pytesseract.image_to_string(Image.fromarray(gray), lang=lang)


def run_ocr_words(gray: np.ndarray, lang: str) -> list[Word]:
    """Return Tesseract's words with their bounding boxes.

    Each word is ``{"text", "box": [x0, y0, x1, y1]}`` in image pixels — the same
    shape the dataset and ML pickers use. Non-text rows (confidence ``-1``) and
    blank tokens are dropped.
    """
    data = pytesseract.image_to_data(
        Image.fromarray(gray), lang=lang, output_type=pytesseract.Output.DICT
    )
    words: list[Word] = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        x, y = int(data["left"][i]), int(data["top"][i])
        w, h = int(data["width"][i]), int(data["height"][i])
        words.append({"text": text, "box": [x, y, x + w, y + h]})
    return words
