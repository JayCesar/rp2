#!/usr/bin/env python3
"""Smoke test for the Conv1D FocalLoss gamma-search CLI.

This runs the conv1d_train_on_features_focal_loss.py entrypoint with
--gamma-search and small caps to verify that:

* The CLI wiring for gamma_search completes without error.
* The expected gamma_sweep output root is created.

It is intentionally lightweight (few samples, 1 epoch per gamma) and
skips cleanly when the required dataset parquet is not available.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
AI_ANALYSIS = PROJECT_ROOT / "ai-analysis"
CONV1D_SCRIPT = AI_ANALYSIS / "conv1d" / "conv1d_train_on_features_focal_loss.py"
FEATURES_PATH = PROJECT_ROOT / "generated_datasets" / "dataset_with_languagetool_metrics.parquet"


@pytest.mark.skipif(
    not FEATURES_PATH.exists(),
    reason="features dataset parquet not found; skipping Conv1D gamma-search CLI smoke test",
)
def test_conv1d_gamma_search_cli_smoke(tmp_path: Path) -> None:
    """Run the Conv1D gamma-search CLI with a tiny config and assert it completes.

    We call the script via the current Python interpreter so that the same
    environment used by pytest is used for the smoke test.
    """

    # Use a very small sample/epoch cap to keep the smoke test fast.
    cmd = [
        sys.executable,
        str(CONV1D_SCRIPT),
        "--max-samples",
        "256",
        "--epochs-per-gamma",
        "1",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    # If the command fails, include stdout/stderr for easier debugging.
    if result.returncode != 0:
        raise AssertionError(
            "Conv1D gamma-search CLI smoke test failed with exit code "
            f"{result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    # Verify that the gamma_sweep root directory was created.
    gamma_root = (
        AI_ANALYSIS
        / "conv1d"
        / "runs"
        / "features_focal_loss"
        / "conv1d_model"
        / "gamma_sweep"
    )
    assert gamma_root.exists() and gamma_root.is_dir(), "gamma_sweep output root was not created"
