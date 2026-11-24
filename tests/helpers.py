"""Helpers to synthesise receipt photos for tests."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def make_receipt_photo(
    *,
    angle: float = 14.0,
    canvas: tuple[int, int] = (900, 1200),
    receipt: tuple[int, int] = (360, 720),
    background: int = 45,
    fmt: str = "JPEG",
) -> bytes:
    """Render a bright, portrait receipt tilted on a darker background.

    Returns encoded image bytes. The receipt carries a few dark "text" lines so
    the result is clearly grayscale rather than a flat rectangle.
    """
    cw, ch = canvas
    rw, rh = receipt
    bg = Image.new("RGB", (cw, ch), (background, background, background))

    paper = Image.new("RGB", (rw, rh), (250, 250, 248))
    draw = ImageDraw.Draw(paper)
    # Store header.
    draw.rectangle([30, 30, rw - 30, 70], fill=(20, 20, 20))
    # Body text rows.
    for i in range(6):
        y = 120 + i * 60
        draw.rectangle([30, y, rw - 60, y + 18], fill=(40, 40, 40))
    # Total row.
    draw.rectangle([30, rh - 120, rw - 30, rh - 80], fill=(15, 15, 15))

    rotated = paper.rotate(angle, expand=True, fillcolor=(background, background, background))
    px = (cw - rotated.width) // 2
    py = (ch - rotated.height) // 2
    # Paste only the receipt pixels (mask out the background fill) so the tilt
    # is preserved without a surrounding bright box.
    mask = Image.new("L", rotated.size, 0)
    mdraw = ImageDraw.Draw(mask)
    solid = Image.new("L", paper.size, 255).rotate(angle, expand=True, fillcolor=0)
    mask.paste(solid, (0, 0))
    bg.paste(rotated, (px, py), mask)
    return _encode(bg, fmt)


def make_receipt_on_light_background(
    *,
    angle: float = 12.0,
    canvas: tuple[int, int] = (900, 1200),
    receipt: tuple[int, int] = (360, 720),
    background: int = 225,
    fmt: str = "JPEG",
) -> bytes:
    """Render a tilted receipt on a *bright* table, close in brightness to paper.

    The paper (250) sits only a little brighter than the background (default
    225), so the "largest bright region" heuristic cannot separate them. A soft
    drop shadow rings the paper — as a real receipt on a bright desk casts —
    giving the edge that OpenCV edge detection relies on.
    """
    cw, ch = canvas
    rw, rh = receipt
    bg = Image.new("RGB", (cw, ch), (background, background, background))

    paper = Image.new("RGB", (rw, rh), (250, 250, 248))
    draw = ImageDraw.Draw(paper)
    draw.rectangle([30, 30, rw - 30, 70], fill=(20, 20, 20))
    for i in range(6):
        y = 120 + i * 60
        draw.rectangle([30, y, rw - 60, y + 18], fill=(40, 40, 40))
    draw.rectangle([30, rh - 120, rw - 30, rh - 80], fill=(15, 15, 15))

    rotated = paper.rotate(angle, expand=True, fillcolor=(background, background, background))
    solid = Image.new("L", paper.size, 255).rotate(angle, expand=True, fillcolor=0)
    px = (cw - rotated.width) // 2
    py = (ch - rotated.height) // 2

    # Soft shadow halo around the whole paper so there is a clear edge on a
    # bright table even though paper and table are close in brightness.
    shadow_mask = solid.filter(ImageFilter.GaussianBlur(12))
    bg.paste((140, 140, 140), (px, py + 6), shadow_mask)
    bg.paste(rotated, (px, py), solid)
    return _encode(bg, fmt)


def make_text_receipt_photo(
    lines: list[str],
    *,
    angle: float = 12.0,
    canvas: tuple[int, int] = (900, 1200),
    receipt: tuple[int, int] = (480, 760),
    background: int = 45,
    fmt: str = "JPEG",
) -> bytes:
    """Render a receipt carrying real text lines, tilted on a dark background.

    The text is drawn with a scalable bundled font so Tesseract can read it back
    after the scan pipeline crops and straightens the receipt.
    """
    cw, ch = canvas
    rw, rh = receipt
    paper = Image.new("RGB", (rw, rh), (250, 250, 248))
    draw = ImageDraw.Draw(paper)
    font = ImageFont.load_default(size=30)
    y = 40
    for line in lines:
        draw.text((30, y), line, fill=(15, 15, 15), font=font)
        y += 70

    bg = Image.new("RGB", (cw, ch), (background, background, background))
    rotated = paper.rotate(angle, expand=True, fillcolor=(background, background, background))
    mask = Image.new("L", paper.size, 255).rotate(angle, expand=True, fillcolor=0)
    px = (cw - rotated.width) // 2
    py = (ch - rotated.height) // 2
    bg.paste(rotated, (px, py), mask)
    return _encode(bg, fmt)


def make_plain_photo(
    *, size: tuple[int, int] = (640, 480), color: int = 130, fmt: str = "JPEG"
) -> bytes:
    """A flat image with no distinct bright region (no receipt to find)."""
    img = Image.new("RGB", size, (color, color, color))
    # Add a little noise so it is not a degenerate single-value image.
    arr = np.asarray(img, dtype=np.int16)
    rng = np.random.default_rng(0)
    arr = np.clip(arr + rng.integers(-8, 8, arr.shape), 0, 255).astype(np.uint8)
    return _encode(Image.fromarray(arr, "RGB"), fmt)


def _encode(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def fake_heic_bytes() -> bytes:
    """Minimal bytes that look like a HEIC file by its ISO-BMFF ftyp brand."""
    return b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heic" + b"\x00" * 64
