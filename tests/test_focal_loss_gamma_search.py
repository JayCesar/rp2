#!/usr/bin/env python3
"""Gamma search tests for FocalLoss across Conv1D models, future-ready for BLSTM.

- Compares candidate gamma values and selects best by QWK, then Kappa, then MAE (tie-breakers).
- Provides a reusable run_gamma_search() API that accepts model-agnostic factories.
- Includes tests for Conv1D on features and vectorized essays, and a future BLSTM stub.

Notes:
- Uses tiny configs, CPU-only, 1 epoch per gamma, and small sample caps to keep tests fast.
- Skips cleanly when datasets are not present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple, Any
import math
import random

import numpy as np
import pytest
import torch
import polars as pl
import polars.selectors as cs

# Ensure ai-analysis is importable
import sys

PROJECT_ROOT = Path(__file__).parent.parent
AI_ANALYSIS = PROJECT_ROOT / "ai-analysis"
if str(AI_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(AI_ANALYSIS))
if str(AI_ANALYSIS / "conv1d") not in sys.path:
    sys.path.insert(0, str(AI_ANALYSIS / "conv1d"))

# Common utilities
from common import (
    EssayDataset,
    TargetScaler,
    create_data_loader,
    split_dataset,
    set_seed,
)

# Conv1D focal loss training pieces
from conv1d import ModelConfig, TrainConfig, SerializationConfig
from conv1d_focal_loss import Conv1DClassifier, validate_scores_for_ce
from conv1d_trainer_focal_loss import Conv1DFLTrainer, calculate_alpha_from_frequency

# Focal Loss
from common.focal_loss import FocalLoss


# ----------------------------
# Reusable gamma-search helper
# ----------------------------
MetricDict = Dict[str, float]


def _metric_key_for_selection(metrics: MetricDict) -> Tuple[float, float, float]:
    """Return a tuple used to rank results by preference: higher QWK, higher Kappa, lower MAE.
    We negate QWK/Kappa for min() selection convenience.
    """
    qwk = metrics.get("qwk", float("nan"))
    kappa = metrics.get("kappa", float("nan"))
    mae = metrics.get("mae", float("nan"))
    # Handle nans by pushing them to worst
    qwk = qwk if math.isfinite(qwk) else -1.0
    kappa = kappa if math.isfinite(kappa) else -1.0
    mae = mae if math.isfinite(mae) else float("inf")
    return (-qwk, -kappa, mae)


def run_gamma_search(
    *,
    trainer_factory: Callable[[Path, Any, Any, Any], Any],
    dataloaders_factory: Callable[[int], Tuple[Any, Any, Any, Any]],
    gamma_values: Sequence[float],
    num_classes: int,
    tmp_dir: Path,
    seed: int = 42,
) -> Tuple[Dict[float, MetricDict], float, MetricDict]:
    """Run a fast gamma sweep and select best gamma by QWK, then Kappa, then MAE.

    Returns (results_by_gamma, best_gamma, best_metrics).
    """
    # Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    # Small sample cap for speed
    max_samples = 400

    # Build loaders and class frequencies (dataset/train_df provided by factory)
    train_df, train_loader, val_loader, test_loader = dataloaders_factory(max_samples)

    class_frequencies = calculate_alpha_from_frequency(
        [
            # derive counts from train_df c1 column
            # map to classes via score/40 and count occurrences
        ]
    )
    # If the factory returns train_df, compute counts properly:
    scores = torch.tensor(train_df["c1"].to_numpy(), dtype=torch.float32)
    class_idxs = (scores / 40).long().tolist()
    counts = [0] * num_classes
    for ci in class_idxs:
        counts[int(ci)] += 1
    alpha = calculate_alpha_from_frequency(counts)

    results: Dict[float, MetricDict] = {}

    for gamma in gamma_values:
        out_dir = tmp_dir / f"gamma_{gamma}"
        out_dir.mkdir(parents=True, exist_ok=True)

        trainer = trainer_factory(out_dir, train_loader, val_loader, test_loader)

        # Override criterion to test this gamma
        trainer.criterion = FocalLoss(
            gamma=gamma,
            alpha=alpha,
            task_type="multi-class",
            num_classes=num_classes,
        )

        metrics = trainer.train()  # should return dict[str, float]
        # Sanity: ensure required keys
        for key in [
            "mae",
            "rmse",
            "qwk",
            "kappa",
            "r2",
            "pearson_corr",
            "step_accuracy",
        ]:
            assert key in metrics, f"Missing metric {key} for gamma={gamma}"
        results[gamma] = metrics

    # Select best gamma by preferred order: QWK desc, Kappa desc, MAE asc
    best_gamma = min(
        results.keys(), key=lambda g: _metric_key_for_selection(results[g])
    )
    best_metrics = results[best_gamma]
    return results, best_gamma, best_metrics


# -------------------------------------
# Conv1D: feature-based training factory
# -------------------------------------
FEATURES_PATH = (
    PROJECT_ROOT / "generated_datasets" / "dataset_with_languagetool_metrics.parquet"
)


def _build_feature_df(max_samples: int) -> pl.DataFrame:
    relevant = [pl.col("c1"), cs.matches(r"^[A-Z0-9_]+$")]
    return (
        pl.scan_parquet(FEATURES_PATH)
        .select(relevant)
        .head(max_samples)
        .drop_nulls()
        .unique()
        .collect()
    )


def dataloaders_factory_features(
    max_samples: int,
) -> Tuple[pl.DataFrame, Any, Any, Any]:
    df = _build_feature_df(max_samples)
    dataset = EssayDataset(df)

    # Validate labels
    validate_scores_for_ce(torch.tensor(df["c1"].to_numpy()))

    # Split
    train_ds, val_ds, test_ds = split_dataset(dataset, seed=42)

    # Minimal configs for speed
    input_dim = len(dataset.feature_cols)
    model_config = ModelConfig(
        conv_filters=[16, 24],
        kernel_sizes=[3, 3],
        input_dim=input_dim,
        dense_neurons=32,
        dropout=0.1,
        pooling="max",
    )
    train_config = TrainConfig(
        epochs=1, batch_size=16, lr=1e-3, early_stopping_patience=999, use_amp=False
    )

    # Loaders (CPU-only, num_workers=0)
    train_loader = create_data_loader(
        train_ds, batch_size=train_config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = create_data_loader(
        val_ds, batch_size=train_config.batch_size, shuffle=False, num_workers=0
    )
    test_loader = create_data_loader(
        test_ds, batch_size=train_config.batch_size, shuffle=False, num_workers=0
    )

    # Return df for alpha computation and loaders; model/trainer built in trainer factory
    return df, train_loader, val_loader, test_loader


def trainer_factory_conv1d_features(
    output_dir: Path, train_loader, val_loader, test_loader
) -> Any:
    # Re-create small model config based on loader dataset properties if needed
    # Access dataset from loader.dataset where possible
    ds = train_loader.dataset
    try:
        input_dim = (
            len(ds.dataset.feature_cols)
            if hasattr(ds, "dataset")
            else len(ds.feature_cols)
        )
    except Exception:
        input_dim = 32

    model_config = ModelConfig(
        conv_filters=[16, 24],
        kernel_sizes=[3, 3],
        input_dim=input_dim,
        dense_neurons=32,
        dropout=0.1,
        pooling="max",
    )
    model = Conv1DClassifier(model_config)

    train_config = TrainConfig(
        epochs=1,
        batch_size=train_loader.batch_size or 16,
        lr=1e-3,
        early_stopping_patience=999,
        use_amp=False,
    )
    serialization = SerializationConfig(
        output_dir=output_dir, save_best_only=False, keep_last_k=1
    )

    target_scaler = TargetScaler("none")
    device = torch.device("cpu")

    trainer = Conv1DFLTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_config=model_config,
        train_config=train_config,
        serialization_config=serialization,
        target_scaler=target_scaler,
        device=device,
        class_frequencies=None,  # we'll override criterion with alpha externally
    )
    return trainer


# -----------------------------------------------
# Conv1D: vectorized-essay training loader factory
# -----------------------------------------------
VECTORS_PATH = (
    PROJECT_ROOT
    / "generated_datasets"
    / "extended_essay-br_preprocessed_for_BLSTM.parquet"
)


def _build_vectorized_df(max_samples: int) -> pl.DataFrame:
    try:
        relevant = ("c1", "essay_token_embeddings", "essay_token_length")
        q = pl.scan_parquet(VECTORS_PATH).select(relevant)
    except Exception:
        relevant = ("c1", "essay_token_embeddings")
        q = pl.scan_parquet(VECTORS_PATH).select(relevant)
    return q.head(max_samples).drop_nulls().unique().collect()


def dataloaders_factory_vectorized(
    max_samples: int,
) -> Tuple[pl.DataFrame, Any, Any, Any]:
    df = _build_vectorized_df(max_samples)
    dataset = EssayDataset(df)

    # Validate labels
    validate_scores_for_ce(torch.tensor(df["c1"].to_numpy()))

    # Split
    train_ds, val_ds, test_ds = split_dataset(dataset, seed=42)

    # Minimal configs for speed (input_dim determined by model internally for sequences)
    train_config = TrainConfig(
        epochs=1, batch_size=8, lr=1e-3, early_stopping_patience=999, use_amp=False
    )

    # Loaders (CPU-only, num_workers=0)
    train_loader = create_data_loader(
        train_ds, batch_size=train_config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = create_data_loader(
        val_ds, batch_size=train_config.batch_size, shuffle=False, num_workers=0
    )
    test_loader = create_data_loader(
        test_ds, batch_size=train_config.batch_size, shuffle=False, num_workers=0
    )

    return df, train_loader, val_loader, test_loader


def trainer_factory_conv1d_vectorized(
    output_dir: Path, train_loader, val_loader, test_loader
) -> Any:
    # Use default Conv1DClassifier but with tiny channels for speed
    model_config = ModelConfig(
        conv_filters=[16, 24],
        kernel_sizes=[3, 3],
        input_dim=768,
        dense_neurons=32,
        dropout=0.1,
        pooling="max",
    )
    model = Conv1DClassifier(model_config)

    train_config = TrainConfig(
        epochs=1,
        batch_size=train_loader.batch_size or 8,
        lr=1e-3,
        early_stopping_patience=999,
        use_amp=False,
    )
    serialization = SerializationConfig(
        output_dir=output_dir, save_best_only=False, keep_last_k=1
    )

    target_scaler = TargetScaler("none")
    device = torch.device("cpu")

    trainer = Conv1DFLTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_config=model_config,
        train_config=train_config,
        serialization_config=serialization,
        target_scaler=target_scaler,
        device=device,
        class_frequencies=None,  # we'll override criterion with alpha externally
    )
    return trainer


# ----------------
# Actual test cases
# ----------------


@pytest.mark.skipif(
    not FEATURES_PATH.exists(),
    reason="features dataset parquet not found; skipping gamma search test",
)
def test_gamma_search_conv1d_features(tmp_path: Path):
    gamma_values = [0.5, 1.0, 2.0, 3.5, 5.0]
    num_classes = 6

    results, best_gamma, best_metrics = run_gamma_search(
        trainer_factory=trainer_factory_conv1d_features,
        dataloaders_factory=dataloaders_factory_features,
        gamma_values=gamma_values,
        num_classes=num_classes,
        tmp_dir=tmp_path,
    )

    # Validate result structure
    for g in gamma_values:
        assert g in results
        metrics = results[g]
        for key in [
            "mae",
            "rmse",
            "qwk",
            "kappa",
            "r2",
            "pearson_corr",
            "step_accuracy",
        ]:
            assert key in metrics
            assert math.isfinite(float(metrics[key]))

    # Selection by QWK desc, then Kappa desc, then MAE asc
    chosen = min(results.keys(), key=lambda g: _metric_key_for_selection(results[g]))
    assert best_gamma == chosen
    assert best_metrics is results[best_gamma]

    # Directories created
    for g in gamma_values:
        assert (tmp_path / f"gamma_{g}").exists()


@pytest.mark.skipif(
    not VECTORS_PATH.exists(),
    reason="vectorized dataset parquet not found; skipping gamma search test",
)
def test_gamma_search_conv1d_vectorized_essays(tmp_path: Path):
    gamma_values = [0.0, 1.0, 2.0, 3.0]
    num_classes = 6

    results, best_gamma, best_metrics = run_gamma_search(
        trainer_factory=trainer_factory_conv1d_vectorized,
        dataloaders_factory=dataloaders_factory_vectorized,
        gamma_values=gamma_values,
        num_classes=num_classes,
        tmp_dir=tmp_path,
    )

    for g in gamma_values:
        assert g in results
        metrics = results[g]
        for key in [
            "mae",
            "rmse",
            "qwk",
            "kappa",
            "r2",
            "pearson_corr",
            "step_accuracy",
        ]:
            assert key in metrics
            assert math.isfinite(float(metrics[key]))

    chosen = min(results.keys(), key=lambda g: _metric_key_for_selection(results[g]))
    assert best_gamma == chosen
    assert best_metrics is results[best_gamma]

    for g in gamma_values:
        assert (tmp_path / f"gamma_{g}").exists()


def test_gamma_search_api_future_blstm(tmp_path: Path):
    # Future stub: when BLSTM FocalLoss exists, plug its factories here.
    pytest.skip("BLSTM FocalLoss trainer not available yet; placeholder test")
