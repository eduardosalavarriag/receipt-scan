"""The one on-disk dataset format shared by download, train and evaluate.

A dataset lives under a data directory (``/data`` in the container) as::

    images/<id>.jpg     the receipt image
    labels/<id>.json    {"source", "company", "total", "words": [{text, box}]}

``box`` is an axis-aligned ``[x0, y0, x1, y1]`` rectangle in image pixels.
``total`` is an exact decimal string with a dot (e.g. ``"51.76"``) or ``null``.
``source`` records where the example came from: ``"sroie"``, ``"cord"``,
``"synthetic"`` (generated German receipts) or ``"labelled"`` (the /label page).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

IMAGES_DIRNAME = "images"
LABELS_DIRNAME = "labels"


class Word(TypedDict):
    text: str
    box: list[int]


class Label(TypedDict):
    source: str
    company: str | None
    total: str | None
    words: list[Word]


def images_dir(data_dir: Path | str) -> Path:
    return Path(data_dir) / IMAGES_DIRNAME


def labels_dir(data_dir: Path | str) -> Path:
    return Path(data_dir) / LABELS_DIRNAME


def label_path(data_dir: Path | str, example_id: str) -> Path:
    return labels_dir(data_dir) / f"{example_id}.json"


def example_exists(data_dir: Path | str, example_id: str) -> bool:
    """True when a label already exists for this id (writing is idempotent)."""
    return label_path(data_dir, example_id).exists()


def write_example(
    data_dir: Path | str,
    example_id: str,
    image_bytes: bytes,
    label: Label,
    *,
    image_ext: str = "jpg",
) -> bool:
    """Write one example in the dataset format.

    Returns ``True`` if written, ``False`` if an example with this id already
    exists (so re-running ``download`` never duplicates receipts). The label is
    written last, so its presence means the image is there too.
    """
    if example_exists(data_dir, example_id):
        return False
    images = images_dir(data_dir)
    labels = labels_dir(data_dir)
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    (images / f"{example_id}.{image_ext}").write_bytes(image_bytes)
    label_path(data_dir, example_id).write_text(
        json.dumps(label, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return True


def iter_labels(data_dir: Path | str) -> Iterator[tuple[str, Label]]:
    """Yield ``(id, label)`` for every label in the data directory, sorted."""
    labels = labels_dir(data_dir)
    if not labels.exists():
        return
    for path in sorted(labels.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        yield path.stem, data


def count_examples(data_dir: Path | str) -> int:
    labels = labels_dir(data_dir)
    if not labels.exists():
        return 0
    return sum(1 for _ in labels.glob("*.json"))
