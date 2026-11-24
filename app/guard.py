"""The accuracy guard (#10): fail CI when total accuracy regresses.

CI runs this over a small, fixed CORD sample committed under ``ci/cord-sample``
and fails the build when the extractor's **total** accuracy drops below the
recorded baseline. A change that makes reading worse is flagged before merge.

The baseline lives beside the sample in ``baseline.json`` (``{"total_accuracy":
...}``), so raising or lowering it is a deliberate, reviewable diff. The guard
reads the sample's stored OCR words only — no images, no network — so it is fully
deterministic and runs in CI without fetching anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .evaluate import EvaluationResult, Predictor, evaluate_data_dir, format_report, rules_predict

# The committed fixed CORD sample and its baseline (repo-root relative).
DEFAULT_SAMPLE_DIR = Path(__file__).resolve().parent.parent / "ci" / "cord-sample"
BASELINE_FILENAME = "baseline.json"

# Float slack so an unchanged extractor is never flagged by rounding alone.
_EPSILON = 1e-9


@dataclass
class GuardResult:
    """Outcome of one guard run: the baseline, what we measured, the full report."""

    baseline: float
    measured: float
    evaluation: EvaluationResult

    @property
    def passed(self) -> bool:
        """True when measured total accuracy is at or above the baseline."""
        return self.measured + _EPSILON >= self.baseline


def baseline_path(sample_dir: Path | str) -> Path:
    return Path(sample_dir) / BASELINE_FILENAME


def load_baseline(sample_dir: Path | str) -> float:
    """Read the recorded baseline total accuracy from the sample's baseline.json."""
    data = json.loads(baseline_path(sample_dir).read_text(encoding="utf-8"))
    return float(data["total_accuracy"])


def run_guard(
    sample_dir: Path | str = DEFAULT_SAMPLE_DIR, predict: Predictor = rules_predict
) -> GuardResult:
    """Score ``predict`` on the sample and compare total accuracy to the baseline."""
    baseline = load_baseline(sample_dir)
    evaluation = evaluate_data_dir(sample_dir, predict=predict)
    return GuardResult(
        baseline=baseline, measured=evaluation.total.accuracy, evaluation=evaluation
    )


def format_guard(result: GuardResult) -> str:
    """A human-readable guard summary: the full report, then pass/fail against baseline."""
    status = "PASS" if result.passed else "FAIL"
    lines = [
        format_report(result.evaluation),
        "",
        f"Baseline total accuracy: {result.baseline * 100:.1f}%",
        f"Measured total accuracy: {result.measured * 100:.1f}%",
        f"Accuracy guard: {status}",
    ]
    if not result.passed:
        lines.append(
            "Total accuracy dropped below the baseline. Fix the regression, or, "
            "if the drop is intended, update total_accuracy in "
            f"{BASELINE_FILENAME}."
        )
    return "\n".join(lines)
