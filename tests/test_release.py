"""Release artefacts: the NOTICE file, the ghcr.io release workflow and the
README sections that document the published image (issue #9).

These are text-level checks so they stay dependency-free and deterministic in
CI: they never build an image or reach the network.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE = "ghcr.io/eduardosalavarriag/receipt-scan"


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# NOTICE
# ---------------------------------------------------------------------------
def test_notice_credits_datasets_and_tesseract() -> None:
    notice = _read("NOTICE")
    assert "SROIE" in notice
    assert "ICDAR 2019" in notice
    assert "CORD" in notice
    assert "NAVER Clova" in notice
    assert "Tesseract" in notice


# ---------------------------------------------------------------------------
# Release workflow
# ---------------------------------------------------------------------------
def test_release_workflow_publishes_versioned_and_latest_to_ghcr() -> None:
    wf = _read(".github/workflows/release.yml")
    # Triggered by version tags.
    assert "tags:" in wf
    assert "v" in wf
    # Publishes the versioned and the latest tag to ghcr.io.
    assert IMAGE in wf
    assert "latest" in wf
    # Authenticates to ghcr with the Actions token and needs write permission.
    assert "ghcr.io" in wf
    assert "GITHUB_TOKEN" in wf or "github.token" in wf
    assert "packages: write" in wf


def test_release_workflow_bakes_a_trained_model_into_the_image() -> None:
    wf = _read(".github/workflows/release.yml")
    # The model is trained from downloaded data, then placed where the image
    # copies it in (app/builtin_model.joblib, see app.model.BUILTIN_MODEL_PATH).
    assert "python -m app download" in wf
    assert "python -m app train" in wf
    assert "app/builtin_model.joblib" in wf


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------
def test_readme_documents_pulling_the_published_image() -> None:
    readme = _read("README.md")
    assert IMAGE in readme
    # The published image serves with the built-in model, no /data mount.
    assert "built-in model" in readme.lower()


def test_readme_documents_adding_a_tesseract_language() -> None:
    readme = _read("README.md").lower()
    assert "adding a tesseract language" in readme
    # Points at the apt package naming for Tesseract language data.
    assert "tesseract-ocr-" in readme


def test_readme_documents_core_commands_and_settings() -> None:
    readme = _read("README.md")
    for command in ("download", "label", "train", "evaluate"):
        assert command in readme
    for setting in ("OCR_LANG", "API_TOKEN", "LABELING", "S3_BUCKET"):
        assert setting in readme
