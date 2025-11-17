#!/usr/bin/env python3
"""Smoke tests for BLSTM FocalLoss gamma-search CLIs.

These tests mirror the Conv1D gamma-search CLI smoke tests but for BLSTM:
- Verify the BLSTM FocalLoss gamma-sweep scripts run without error.
- Assert that the expected gamma_sweep output roots are created.

Tests skip cleanly when the required datasets are not present.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
AI_ANALYSIS = PROJECT_ROOT / "ai-analysis"

BLSTM_FEATURES_SCRIPT = AI_ANALYSIS / "blstm" / "blstm_train_on_features_focal_loss.py"
BLSTM_VECTOR_SCRIPT = AI_ANALYSIS / "blstm" / "blstm_train_on_vectorized_essays_focal_loss.py"

FEATURES_PATH = PROJECT_ROOT / "generated_datasets" / "dataset_with_languagetool_metrics.parquet"
VECTORS_FUSED = (
    PROJECT_ROOT
    / "generated_datasets"
    / "extended_essay-br_preprocessed_for_BLSTM.parquet"
)
VECTORS_P1 = (
    PROJECT_ROOT
    / "generated_datasets"
    / "extended_essay-br_preprocessed_for_BLSTM_part1.parquet"
)
VECTORS_P2 = (
    PROJECT_ROOT
    / "generated_datasets"
    / "extended_essay-br_preprocessed_for_BLSTM_part2.parquet"
)


@ pytest.mark.skipif(
    not FEATURES_PATH.exists(),
    reason="features dataset parquet not found; skipping BLSTM features gamma-search CLI smoke test",
)
def test_blstm_gamma_search_features_cli_smoke(tmp_path: Path) -> None:
    """Run the BLSTM features gamma-search CLI with a tiny config."""

    cmd = [
        sys.executable,
        str(BLSTM_FEATURES_SCRIPT),
        "--max-samples",
        "256",
        "--epochs-per-gamma",
        "1",
        "--gamma-grid",
        "1",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise AssertionError(
            "BLSTM features gamma-search CLI smoke test failed with exit code "
            f"{result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    gamma_root = (
        AI_ANALYSIS
        / "blstm"
        / "runs"
        / "features_focal_loss"
        / "blstm_model"
        / "gamma_sweep"
    )
    assert gamma_root.exists() and gamma_root.is_dir(), "gamma_sweep output root was not created for BLSTM features"


@ pytest.mark.skipif(
    not (VECTORS_FUSED.exists() or (VECTORS_P1.exists() and VECTORS_P2.exists())),
    reason=(
        "BLSTM fused vectorized dataset (or its parts) not found; skipping BLSTM "
        "vectorized gamma-search CLI smoke test"
    ),
)
def test_blstm_gamma_search_vectorized_cli_smoke(tmp_path: Path) -> None:
    """Run the BLSTM vectorized-essays gamma-search CLI with a tiny config."""

    cmd = [
        sys.executable,
        str(BLSTM_VECTOR_SCRIPT),
        "--max-samples",
        "256",
        "--epochs-per-gamma",
        "1",
        "--gamma-grid",
        "1",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise AssertionError(
            "BLSTM vectorized gamma-search CLI smoke test failed with exit code "
            f"{result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    gamma_root = (
        AI_ANALYSIS
        / "blstm"
        / "runs"
        / "vectorized_essays_focal_loss"
        / "blstm_model"
        / "gamma_sweep"
    )
    assert gamma_root.exists() and gamma_root.is_dir(), "gamma_sweep output root was not created for BLSTM vectorized essays"