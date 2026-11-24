from __future__ import annotations

import json
from pathlib import Path

from app.__main__ import main
from app.dataset import Label, Word
from app.guard import DEFAULT_SAMPLE_DIR, run_guard


def _write_sample(sample_dir: Path, baseline: float) -> None:
    """Write one well-formed CORD-style example and a baseline.json."""
    labels = sample_dir / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    words: list[Word] = [
        {"text": "SHOP", "box": [10, 10, 100, 40]},
        {"text": "TOTAL 7.50", "box": [10, 60, 200, 90]},
    ]
    label: Label = {"source": "cord", "company": "SHOP", "total": "7.50", "words": words}
    (labels / "cord-1.json").write_text(json.dumps(label), encoding="utf-8")
    (sample_dir / "baseline.json").write_text(
        json.dumps({"total_accuracy": baseline}), encoding="utf-8"
    )


def _wrong_predictor(words: list[Word]) -> tuple[str | None, str | None]:
    """A regressed extractor that never reads the total."""
    return None, None


# ---------------------------------------------------------------------------
# pass / fail against the baseline
# ---------------------------------------------------------------------------
def test_guard_passes_when_accuracy_meets_baseline(tmp_path: Path) -> None:
    _write_sample(tmp_path, baseline=1.0)

    result = run_guard(tmp_path)

    assert result.measured == 1.0
    assert result.passed is True


def test_guard_fails_when_accuracy_below_baseline(tmp_path: Path) -> None:
    # A PR that lowers total accuracy (here, a predictor that reads nothing)
    # falls below the recorded baseline and must fail the guard.
    _write_sample(tmp_path, baseline=1.0)

    result = run_guard(tmp_path, predict=_wrong_predictor)

    assert result.measured == 0.0
    assert result.passed is False


# ---------------------------------------------------------------------------
# the committed fixed CORD sample stays at or above its recorded baseline
# ---------------------------------------------------------------------------
def test_committed_cord_sample_meets_its_baseline() -> None:
    result = run_guard()

    assert result.evaluation.examples >= 1
    assert result.passed is True
    assert result.measured >= result.baseline


def test_default_sample_dir_points_at_committed_sample() -> None:
    assert (DEFAULT_SAMPLE_DIR / "baseline.json").exists()
    assert (DEFAULT_SAMPLE_DIR / "labels").is_dir()


# ---------------------------------------------------------------------------
# CLI exit codes: non-zero on regression, zero on pass
# ---------------------------------------------------------------------------
def test_cli_guard_exit_zero_on_pass(tmp_path: Path) -> None:
    _write_sample(tmp_path, baseline=1.0)

    assert main(["guard", "--sample-dir", str(tmp_path)]) == 0


def test_cli_guard_exit_nonzero_on_regression(tmp_path: Path) -> None:
    # Baseline set above what the rules extractor can reach on this sample.
    _write_sample(tmp_path, baseline=1.0)
    # Add an example the rules get wrong so measured (0.5) < baseline (1.0).
    labels = tmp_path / "labels"
    bad: Label = {
        "source": "cord",
        "company": "MART",
        "total": "2.00",
        "words": [
            {"text": "MART", "box": [10, 10, 100, 40]},
            {"text": "CASH 99.00", "box": [10, 60, 200, 90]},
            {"text": "PAID 2.00", "box": [10, 110, 200, 140]},
        ],
    }
    (labels / "cord-2.json").write_text(json.dumps(bad), encoding="utf-8")

    assert main(["guard", "--sample-dir", str(tmp_path)]) == 1
