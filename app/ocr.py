"""Run Tesseract OCR over a processed receipt image."""

from __future__ import annotations

import numpy as np
import pytesseract
from PIL import Image


def run_ocr(gray: np.ndarray, lang: str) -> str:
    """Return the text Tesseract reads from a grayscale image.

    ``lang`` is a Tesseract language spec such as ``deu+eng`` (from ``OCR_LANG``).
    """
    return pytesseract.image_to_string(Image.fromarray(gray), lang=lang)
