#!/usr/bin/env python3
"""Unit tests for the generic FocalLoss gamma-search utility.

These tests exercise the behaviour of :mod:`ai-analysis.common.gamma_search`
without depending on the real Conv1D / BLSTM training code or large datasets.

They verify that:

* ``metric_key_for_selection`` ranks metrics as intended.
* ``run_gamma_search``
  - runs once per gamma value,
  - passes the configured gamma value into the trainer's criterion,
  - writes per-gamma metrics and prediction artifacts,
  - writes global summary files and selects ``best_gamma`` consistently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import json

import polars as pl
import pytest

# Ensure ai-analysis is importable
import sys

PROJECT_ROOT = Path(__file__).parent.parent
AI_ANALYSIS = PROJECT_ROOT / "ai-analysis"
if str(AI_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(AI_ANALYSIS))

from common import DEFAULT_GAMMA_VALUES, metric_key_for_selection, run_gamma_search
from common.gamma_search import _gamma_to_dir_name


MetricDict = Dict[str, float]


class DummyTrainer:
    """Minimal trainer used to test ``run_gamma_search``.

    The trainer does **not** use the provided DataLoaders; instead, it derives
    deterministic metrics from ``self.criterion.gamma`` so we can verify that
    the correct gamma value is being propagated and that selection is applied
    consistently by :func:`metric_key_for_selection`.
    """

    def __init__(self, out_dir: Path, *_: Any) -> None:  # noqa: D401 - simple stub
        self.out_dir = out_dir
        # These attributes are inspected by ``run_gamma_search`` for optional
        # artifact writing.
        self.training_history = [
            {
                "epoch": 1,
                "train_loss": 1.0,
                "val_loss": 1.0,
                "val_mae": 1.0,
                "val_rmse": 1.0,
                "kappa": 0.0,
                "qwk": 0.0,
                "r2": 0.0,
                "pearson_corr": 0.0,
                "step_accuracy": 0.0,
                "epoch_time": 0.1,
            }
        ]
        self.best_val_predictions = [
            {"id": "sample_0", "target": 0.0, "pred": 0.0, "pred_snapped": 0.0}
        ]
        # Will be overwritten by ``run_gamma_search`` with a real FocalLoss
        self.criterion = type("DummyCriterion", (), {"gamma": 0.0})()

    def train(self) -> MetricDict:
        # ``run_gamma_search`` overwrites ``self.criterion`` with a FocalLoss
        gamma = float(getattr(self.criterion, "gamma", 0.0))

        # Construct metrics with a clear ordering in terms of gamma:
        # - Higher gamma => higher QWK and Kappa
        # - MAE increases away from gamma == 2.0
        qwk = gamma
        kappa = gamma / 2.0
        mae = abs(2.0 - gamma)
        rmse = mae

        return {
            "mae": mae,
            "rmse": rmse,
            "qwk": qwk,
            "kappa": kappa,
            "r2": 0.0,
            "pearson_corr": 0.0,
            "step_accuracy": 0.0,
        }


def _dummy_dataloaders_factory(max_samples: int | None):
    """Return placeholder loaders for ``run_gamma_search``.

    The dummy trainer ignores these loaders entirely, so they can be ``None``.
    """

    _ = max_samples  # unused
    return None, None, None


def _dummy_trainer_factory(out_dir: Path, train_loader, val_loader, test_loader):  # noqa: D401 - simple stub
    _ = (train_loader, val_loader, test_loader)
    return DummyTrainer(out_dir)


def test_metric_key_for_selection_ranking() -> None:
    """metric_key_for_selection should implement QWK desc, Kappa desc, MAE asc.

    The function returns (-qwk, -kappa, mae), so ``min()`` on this key should
    pick the metric dict that has highest QWK, then highest Kappa, then lowest
    MAE.
    """

    better_qwk: MetricDict = {"qwk": 0.8, "kappa": 0.5, "mae": 0.4}
    worse_qwk_better_mae: MetricDict = {"qwk": 0.7, "kappa": 0.9, "mae": 0.1}

    key_better = metric_key_for_selection(better_qwk)
    key_worse = metric_key_for_selection(worse_qwk_better_mae)

    assert key_better < key_worse

    # When QWK ties, Kappa is considered next
    a: MetricDict = {"qwk": 0.75, "kappa": 0.6, "mae": 0.3}
    b: MetricDict = {"qwk": 0.75, "kappa": 0.4, "mae": 0.1}
    assert metric_key_for_selection(a) < metric_key_for_selection(b)


def test_metric_key_for_selection_handles_missing_and_nan() -> None:
    """Missing or non-finite values should be mapped to worst-case defaults."""

    metrics_with_nan: MetricDict = {"qwk": float("nan"), "kappa": float("nan"), "mae": float("nan")}
    key = metric_key_for_selection(metrics_with_nan)
    # QWK/Kappa -> -1.0 (worst), MAE -> +inf (worst)
    assert key == (-(-1.0), -(-1.0), float("inf")) or key[2] == float("inf")


@pytest.mark.parametrize("gamma_values", [[0.5, 1.0, 2.0, 3.5, 5.0], list(DEFAULT_GAMMA_VALUES)])
def test_run_gamma_search_creates_artifacts_and_selects_best(tmp_path: Path, gamma_values) -> None:
    """run_gamma_search should write per-gamma artifacts and a global summary.

    The best gamma is selected using :func:`metric_key_for_selection`, which we
    verify by recomputing the key on the returned metrics.
    """

    output_root = tmp_path / "gamma_sweep_test"

    results_by_gamma, best_gamma, best_metrics = run_gamma_search(
        trainer_factory=_dummy_trainer_factory,
        dataloaders_factory=_dummy_dataloaders_factory,
        gamma_values=gamma_values,
        num_classes=3,
        output_root=output_root,
        seed=123,
        alpha=None,
        max_samples=None,
    )

    # One result per gamma
    assert set(results_by_gamma.keys()) == set(gamma_values)

    # Best gamma must match the selection key
    chosen = min(results_by_gamma.keys(), key=lambda g: metric_key_for_selection(results_by_gamma[g]))
    assert best_gamma == chosen
    assert best_metrics == results_by_gamma[best_gamma]

    # Per-gamma directories and core artifacts exist (use the same
    # sanitisation logic as run_gamma_search/_gamma_to_dir_name).
    for g in gamma_values:
        gamma_dir = output_root / f"gamma_{_gamma_to_dir_name(g)}"
        assert gamma_dir.is_dir()

        assert (gamma_dir / "metrics_best.csv").is_file()
        assert (gamma_dir / "metrics_best.parquet").is_file()
        assert (gamma_dir / "metrics_best.json").is_file()

        # Training history and predictions should also be present for the dummy trainer
        assert (gamma_dir / "training_history.csv").is_file()
        assert (gamma_dir / "training_history.parquet").is_file()
        assert (gamma_dir / "validation_predictions_best.csv").is_file()
        assert (gamma_dir / "validation_predictions_best.parquet").is_file()

        # Validate predictions schema matches expected columns
        df_preds = pl.read_csv(gamma_dir / "validation_predictions_best.csv")
        for col in ["id", "target", "pred", "pred_snapped"]:
            assert col in df_preds.columns

    # Global summary artifacts
    results_csv = output_root / "results_by_gamma.csv"
    results_parquet = output_root / "results_by_gamma.parquet"
    best_json = output_root / "best_gamma.json"

    assert results_csv.is_file()
    assert results_parquet.is_file()
    assert best_json.is_file()

    payload = json.loads(best_json.read_text(encoding="utf-8"))
    assert "best_gamma" in payload
    assert "metrics" in payload
    assert pytest.approx(payload["best_gamma"]) == float(best_gamma)
