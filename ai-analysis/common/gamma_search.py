"""Generic FocalLoss gamma sweep utilities.

This module provides a small, model-agnostic helper to run sequential
FocalLoss gamma sweeps for any trainer that exposes a minimal interface:

- ``trainer.criterion`` is an nn.Module that can be replaced.
- ``trainer.train()`` returns a ``dict[str, float]`` with evaluation metrics
  computed on the validation set (at least ``mae``, ``qwk``, ``kappa`` when
  available).
- Optionally, ``trainer.best_val_predictions`` contains per-sample validation
  predictions as a list[dict] or a Polars DataFrame with at least the
  following columns, matching existing Conv1D conventions::

      id,target,pred,pred_snapped

- Optionally, ``trainer.training_history`` is a list[dict[str, float]]
  describing per-epoch metrics.

Selection rule
--------------

Gamma selection follows the same policy originally implemented in
``tests/test_focal_loss_gamma_search.py``:

1. Prefer **higher QWK** (Quadratic Weighted Kappa).
2. Break ties by **higher Cohen's Kappa**.
3. Break remaining ties by **lower MAE**.

To make this compatible with ``min()`` we turn metrics into a key
``(-qwk, -kappa, mae)``; the gamma producing the smallest key is selected
as the best.

Reusability
-----------

This utility is intentionally unaware of Conv1D or BLSTM specifics.
Future BLSTM FocalLoss training can plug in without any changes by
providing two factories:

- ``dataloaders_factory(max_samples: int | None) -> tuple[train_loader, val_loader, test_loader | None]``
- ``trainer_factory(out_dir: Path, train_loader, val_loader, test_loader | None) -> Trainer``

Example (sketch)::

    from pathlib import Path
    from ai_analysis.common.gamma_search import run_gamma_search

    def dataloaders_factory_blstm(max_samples: int | None = None):
        # Build dataset and DataLoaders for BLSTM here
        return train_loader, val_loader, test_loader

    def trainer_factory_blstm_fl(out_dir: Path, train_loader, val_loader, test_loader):
        # Instantiate BLSTM FocalLoss trainer here
        return trainer

    results_by_gamma, best_gamma, best_metrics = run_gamma_search(
        trainer_factory=trainer_factory_blstm_fl,
        dataloaders_factory=dataloaders_factory_blstm,
        gamma_values=None,   # use default grid
        num_classes=6,
        output_root=Path("ai-analysis/blstm/runs/focal_loss_gamma_search"),
    )

All sweeps are **simple sequential loops over gamma** so they remain
GPU-friendly even for heavier models.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)
import math
import json

import numpy as np
import polars as pl
import torch

from .focal_loss import FocalLoss
from .io_utils import ensure_dir

MetricDict = Dict[str, float]

# Shared default gamma grid (single default for all models)
# Note: 10.0 is included to probe more aggressive focusing behaviour.
DEFAULT_GAMMA_VALUES: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)


def metric_key_for_selection(
    metrics: Mapping[str, float],
) -> Tuple[float, float, float]:
    """Return ranking key ``(-qwk, -kappa, mae)`` for a metrics dict.

    - Higher QWK is better.
    - If QWK ties, higher Cohen's Kappa is better.
    - If both tie, lower MAE is better.

    Missing or non-finite values are treated as worst-case for their
    respective metric (QWK/Kappa -> very low; MAE -> very high).
    """

    def _get(name: str, default: float) -> float:
        value = float(metrics.get(name, default))
        if not math.isfinite(value):
            return default
        return value

    qwk = _get("qwk", -1.0)
    kappa = _get("kappa", -1.0)
    mae = _get("mae", float("inf"))

    # Negate QWK/Kappa so that min() selects best
    return (-qwk, -kappa, mae)


def _gamma_to_dir_name(gamma: float) -> str:
    """Convert a gamma value to a filesystem-safe directory suffix."""

    # Avoid characters like '.' that can be confusing in folder names
    # 0.5 -> "0_5", 2.0 -> "2_0", etc.
    text = f"{gamma:.6g}"  # compact, no trailing zeros where possible
    return text.replace("-", "neg_").replace(".", "_")


def _to_pl_dataframe(rows: Any) -> Optional[pl.DataFrame]:
    """Best-effort conversion of predictions/history to a Polars DataFrame."""

    if rows is None:
        return None
    if isinstance(rows, pl.DataFrame):
        return rows
    if isinstance(rows, list) and rows:
        if isinstance(rows[0], Mapping):
            return pl.DataFrame(rows)  # type: ignore[arg-type]
    return None


def run_gamma_search(
    *,
    trainer_factory: Callable[[Path, Any, Any, Any | None], Any],
    dataloaders_factory: Callable[[Optional[int]], tuple[Any, Any, Any | None]],
    gamma_values: Optional[Sequence[float]],
    num_classes: int,
    output_root: Path,
    seed: int = 42,
    alpha: Optional[Sequence[float] | torch.Tensor] = None,
    max_samples: Optional[int] = None,
) -> Tuple[Dict[float, MetricDict], float, MetricDict]:
    """Run a **sequential** FocalLoss gamma sweep and select the best gamma.

    Args:
        trainer_factory:
            Callable that builds a new trainer instance for a given output
            directory and DataLoaders::

                trainer_factory(out_dir, train_loader, val_loader, test_loader) -> trainer

            ``test_loader`` may be ``None`` if unused.

        dataloaders_factory:
            Callable that prepares DataLoaders (and underlying datasets if
            needed) for training::

                dataloaders_factory(max_samples) -> (train_loader, val_loader, test_loader)

            ``max_samples`` can be used to cap training samples for faster
            sweeps; ``None`` means "use full dataset".

        gamma_values:
            Iterable of candidate gamma values. If ``None``, uses the
            shared default grid :data:`DEFAULT_GAMMA_VALUES`.

        num_classes:
            Number of classes for ``FocalLoss(task_type="multi-class")``.

        output_root:
            Root directory under which per-gamma subdirectories and summary
            files will be written.

        seed:
            Base random seed used via :func:`set_seed` for reproducibility.

        alpha:
            Optional class-weight vector passed to :class:`FocalLoss`. If
            ``None``, FocalLoss is created without explicit class balancing.

        max_samples:
            Optional cap on the number of training samples used when
            constructing DataLoaders.

    Returns:
        ``(results_by_gamma, best_gamma, best_metrics)`` where:

        - ``results_by_gamma`` maps each gamma to its best validation
          metrics dict.
        - ``best_gamma`` is the gamma selected by
          :func:`metric_key_for_selection`.
        - ``best_metrics`` is ``results_by_gamma[best_gamma]``.

    Side effects
    ------------

    For each gamma value, a subdirectory is created under ``output_root``::

        {output_root}/gamma_{gamma_sanitized}/

    Within each subdirectory, the following files are written when data is
    available:

    - ``metrics_best.json`` and ``metrics_best.csv`` – best validation
      metrics for that gamma.
    - ``training_history.csv`` – epoch-wise training/validation history.
    - ``validation_predictions_best.csv`` and
      ``validation_predictions_best.parquet`` – per-sample validation
      predictions, following the same conventions currently used by Conv1D
      (at minimum: ``id,target,pred,pred_snapped``).

    Additionally, two summary files are written at ``output_root``:

    - ``results_by_gamma.csv`` – one row per gamma with its metrics.
    - ``best_gamma.json`` – the selected gamma and its metrics.
    """

    # Prepare output root
    output_root = ensure_dir(output_root)

    # Build loaders once; they are re-used across gamma values
    train_loader, val_loader, test_loader = dataloaders_factory(max_samples)

    if gamma_values is None:
        gamma_values = list(DEFAULT_GAMMA_VALUES)

    # Ensure we have a concrete list for indexing / progress reporting
    gamma_values = list(gamma_values)
    total_gammas = len(gamma_values)

    results_by_gamma: Dict[float, MetricDict] = {}

    # Sequential sweep (no parallelism) to keep GPU/CPU usage simple
    for idx, gamma in enumerate(gamma_values, start=1):
        # Progress message so the user can see which gamma is running
        print(
            f"\n[gamma-search] Starting gamma {idx}/{total_gammas}: gamma={gamma:.6g}"
        )

        gamma_dir_name = f"gamma_{_gamma_to_dir_name(gamma)}"
        out_dir = ensure_dir(output_root / gamma_dir_name)

        # Fresh trainer per gamma
        trainer = trainer_factory(out_dir, train_loader, val_loader, test_loader)

        # Override criterion to test this gamma
        trainer.criterion = FocalLoss(
            gamma=gamma,
            alpha=alpha,
            task_type="multi-class",
            num_classes=num_classes,
        )

        metrics: MutableMapping[str, float] = trainer.train()

        # Print per-gamma best metrics for easier inspection
        print("[gamma-search] Completed gamma " f"{idx}/{total_gammas}: gamma={gamma:.6g}")
        print("[gamma-search] Best validation metrics for this gamma (FocalLoss):")
        for key in sorted(metrics.keys()):
            value = metrics[key]
            try:
                value_str = f"{float(value):.6f}"
            except Exception:
                value_str = str(value)
            print(f"  {key}: {value_str}")

        # Basic sanity: ensure some core metrics exist when possible
        for key in ["mae", "rmse", "qwk", "kappa"]:
            if key not in metrics:
                # Do not fail hard; just skip this check
                continue
            _ = float(metrics[key])  # type: ignore[index]

        results_by_gamma[gamma] = dict(metrics)

        # Persist per-gamma artifacts (include gamma for easier inspection)
        metrics_row: Dict[str, float] = dict(metrics)
        metrics_row["gamma"] = float(gamma)
        df_metrics = pl.DataFrame([metrics_row])
        df_metrics.write_csv(out_dir / "metrics_best.csv")
        df_metrics.write_parquet(out_dir / "metrics_best.parquet")
        (out_dir / "metrics_best.json").write_text(
            json.dumps(metrics_row, indent=2, sort_keys=True), encoding="utf-8"
        )

        # Training history (if any)
        history_df = _to_pl_dataframe(getattr(trainer, "training_history", None))
        if history_df is not None and history_df.height > 0:
            history_df.write_csv(out_dir / "training_history.csv")
            history_df.write_parquet(out_dir / "training_history.parquet")

        # Best validation predictions (if any), following existing conventions
        preds_df = _to_pl_dataframe(getattr(trainer, "best_val_predictions", None))
        if preds_df is not None and preds_df.height > 0:
            preds_df.write_csv(out_dir / "validation_predictions_best.csv")
            preds_df.write_parquet(out_dir / "validation_predictions_best.parquet")

    # Select best gamma using the agreed priority
    best_gamma = min(
        results_by_gamma.keys(),
        key=lambda g: metric_key_for_selection(results_by_gamma[g]),
    )
    best_metrics = results_by_gamma[best_gamma]

    # Global summary: one row per gamma
    rows = []
    for gamma, metrics in results_by_gamma.items():
        row: Dict[str, float] = {"gamma": float(gamma)}
        for k, v in metrics.items():
            try:
                row[k] = float(v)
            except Exception:
                # Skip non-numeric entries
                continue
        row["is_best_gamma"] = float(gamma) == float(best_gamma)
        rows.append(row)

    if rows:
        summary_df = pl.DataFrame(rows)
        summary_df.write_csv(output_root / "results_by_gamma.csv")
        summary_df.write_parquet(output_root / "results_by_gamma.parquet")

    # Write best-gamma metadata and a flat metrics summary at the root
    best_payload = {"best_gamma": float(best_gamma), "metrics": best_metrics}
    (output_root / "best_gamma.json").write_text(
        json.dumps(best_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Convenience: best metrics as a one-row table with gamma column
    best_row: Dict[str, float] = {"gamma": float(best_gamma)}
    for k, v in best_metrics.items():
        try:
            best_row[k] = float(v)
        except Exception:
            continue
    best_df = pl.DataFrame([best_row])
    best_df.write_csv(output_root / "metrics_best_overall.csv")
    best_df.write_parquet(output_root / "metrics_best_overall.parquet")

    return results_by_gamma, best_gamma, best_metrics
