from __future__ import annotations

from pathlib import Path

from app.dataset import (
    Label,
    count_examples,
    example_exists,
    iter_labels,
    write_example,
)


def _label() -> Label:
    return {
        "source": "sroie",
        "company": "TESCO",
        "total": "9.00",
        "words": [{"text": "TESCO", "box": [1, 2, 3, 4]}],
    }


def test_write_example_creates_files(tmp_path: Path) -> None:
    assert write_example(tmp_path, "r1", b"imagebytes", _label()) is True
    assert (tmp_path / "images" / "r1.jpg").read_bytes() == b"imagebytes"
    assert (tmp_path / "labels" / "r1.json").exists()


def test_write_example_is_idempotent(tmp_path: Path) -> None:
    assert write_example(tmp_path, "r1", b"a", _label()) is True
    # Second write with the same id does nothing and keeps the original image.
    assert write_example(tmp_path, "r1", b"different", _label()) is False
    assert (tmp_path / "images" / "r1.jpg").read_bytes() == b"a"
    assert count_examples(tmp_path) == 1


def test_iter_labels_round_trips(tmp_path: Path) -> None:
    write_example(tmp_path, "r1", b"a", _label())
    read = dict(iter_labels(tmp_path))
    assert read["r1"]["company"] == "TESCO"
    assert read["r1"]["words"][0]["text"] == "TESCO"


def test_example_exists_and_empty_dir(tmp_path: Path) -> None:
    assert example_exists(tmp_path, "nope") is False
    assert count_examples(tmp_path) == 0
    assert list(iter_labels(tmp_path)) == []
