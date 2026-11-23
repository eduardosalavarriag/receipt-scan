"""Image decoding and the clean-up pipeline.

The pipeline turns a phone photo of a receipt into a cropped, straightened,
lighting-corrected *grayscale* image. It deliberately does **not** binarise the
result (no black-and-white threshold) so later OCR has the full tonal range.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Image formats we accept from PIL's decoder.
ACCEPTED_PIL_FORMATS = {"JPEG", "PNG", "WEBP", "MPO"}

# ISO-BMFF brand markers that indicate HEIC/HEIF, which PIL cannot decode
# without an extra plugin. We detect them to return a helpful 415.
_HEIF_BRANDS = {b"heic", b"heix", b"heif", b"mif1", b"msf1", b"hevc", b"hevx"}


class DecodeError(Exception):
    """The upload could not be decoded as a supported image."""


class UnsupportedImageError(Exception):
    """The upload is an image, but of an unsupported type (e.g. HEIC)."""


def _looks_like_heif(data: bytes) -> bool:
    # ISO-BMFF: [4-byte size][b"ftyp"][4-byte major brand]...
    if len(data) < 12:
        return False
    if data[4:8] != b"ftyp":
        return False
    return data[8:12] in _HEIF_BRANDS


def decode_image(data: bytes) -> np.ndarray:
    """Decode upload bytes into an RGB uint8 array, honouring EXIF orientation.

    Raises:
        UnsupportedImageError: recognised image type we cannot decode (HEIC).
        DecodeError: not a decodable image at all.
    """
    if _looks_like_heif(data):
        raise UnsupportedImageError(
            "HEIC/HEIF images are not supported. Please export the photo as "
            "JPEG and try again."
        )
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            oriented = ImageOps.exif_transpose(img)
            arr = np.asarray(oriented.convert("RGB"), dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise DecodeError("The upload is not a recognised image.") from exc
    except OSError as exc:
        raise DecodeError("The image could not be read.") from exc

    if fmt is not None and fmt not in ACCEPTED_PIL_FORMATS:
        raise UnsupportedImageError(f"Image format {fmt} is not supported.")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise DecodeError("The image could not be interpreted as RGB.")
    return arr


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left: smallest x+y
    rect[2] = pts[np.argmax(s)]  # bottom-right: largest x+y
    d = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(d)]  # top-right: smallest y-x
    rect[3] = pts[np.argmax(d)]  # bottom-left: largest y-x
    return rect


def find_receipt_quad(gray: np.ndarray) -> np.ndarray | None:
    """Locate the receipt as a 4-point quad, or None if not confident.

    This bootstrap implementation uses the "largest bright region" heuristic:
    threshold the paper away from a darker background and fit a rotated box.
    Issue #3 replaces it with edge detection.
    """
    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    image_area = float(h * w)
    # Require the region to be a meaningful part of the frame, but not the whole
    # frame (that would just be a bright photo, not a receipt on a background).
    if area < 0.05 * image_area or area > 0.98 * image_area:
        return None

    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype(np.float32)
    return _order_corners(box)


def warp_receipt(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-correct the receipt quad to an upright, cropped rectangle."""
    tl, tr, br, bl = quad
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_w = int(max(width_a, width_b))
    max_h = int(max(height_a, height_b))
    if max_w < 2 or max_h < 2:
        return image
    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image, matrix, (max_w, max_h))


def correct_lighting(gray: np.ndarray) -> np.ndarray:
    """Even out uneven lighting and boost local contrast, staying grayscale.

    Divides out a blurred estimate of the background illumination, then applies
    CLAHE. The output keeps a full range of gray values (it is *not* binarised).
    """
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
    background = np.where(background == 0, 1, background).astype(np.float32)
    normalized = (gray.astype(np.float32) / background) * 192.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(normalized)


def process_receipt(rgb: np.ndarray) -> np.ndarray:
    """Run the full pipeline. Returns a single-channel grayscale uint8 image."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    quad = find_receipt_quad(gray)
    if quad is not None:
        gray = warp_receipt(gray, quad)
    return correct_lighting(gray)


def encode_png(gray: np.ndarray) -> bytes:
    """Encode a grayscale image as PNG bytes."""
    ok, buf = cv2.imencode(".png", gray)
    if not ok:
        raise RuntimeError("Failed to encode image as PNG.")
    return buf.tobytes()
