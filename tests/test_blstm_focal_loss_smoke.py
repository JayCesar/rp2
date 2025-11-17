#!/usr/bin/env python3
"""Smoke tests for BiLSTM FocalLoss training integration.

Validates end-to-end integration by running 1 epoch on a tiny feature-based
subset:
- Data loading and preprocessing via EssayDataset
- Model instantiation (BiLSTMClassifier)
- Training loop with FocalLoss (BiLSTMFLTrainer)
- Validation metrics in score space
- Checkpoint and prediction saving with focal metadata
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
import pytest
import torch

# Make ai-analysis and blstm modules importable
PROJECT_ROOT = Path(__file__).parent.parent
AI_ANALYSIS = PROJECT_ROOT / "ai-analysis"

sys.path.insert(0, str(AI_ANALYSIS))
sys.path.insert(0, str(AI_ANALYSIS / "blstm"))

from blstm import (  # type: ignore  # noqa: E402
    ModelConfig,
    SerializationConfig,
    TrainConfig,
    TargetScaler,
    set_seed,
)
from blstm_focal_loss import (  # type: ignore  # noqa: E402
    BiLSTMClassifier,
    validate_scores_for_ce,
)
from trainer_focal_loss import BiLSTMFLTrainer  # type: ignore  # noqa: E402
from common import (  # type: ignore  # noqa: E402
    EssayDataset,
    create_data_loader,
    split_dataset,
)
from common.class_frequencies import (  # type: ignore  # noqa: E402
    get_class_frequencies,
)


project_root = PROJECT_ROOT
logger = logging.getLogger(__name__)


@pytest.fixture
def tiny_dataset():
    """Load a tiny feature-based dataset for BLSTM FL smoke testing."""

    parquet_file = (
        project_root / "generated_datasets" / "dataset_with_languagetool_metrics.parquet"
    )
    if not parquet_file.exists():
        pytest.skip(f"Dataset not found: {parquet_file}")

    relevant_columns = [pl.col("c1"), cs.matches(r"^[A-Z0-9_]+$")]
    dataset_df = (
        pl.scan_parquet(parquet_file)
        .select(relevant_columns)
        .head(500)
        .drop_nulls()
        .unique()
        .collect()
    )
    return EssayDataset(dataset_df)


@pytest.fixture
def smoke_configs(tiny_dataset):
    """Create minimal configs for BLSTM FL smoke testing."""

    model_config = ModelConfig(
        hidden_sizes=[16, 16, 16],
        input_dim=len(tiny_dataset.feature_cols),
        aggregation="attn",
        token_proj_dim=None,
        mlp_hidden=32,
        use_layer_norm=False,
    )

    train_config = TrainConfig(
        epochs=1,
        batch_size=16,
        lr=1e-3,
        optimizer="adamw",
        early_stopping_patience=999,
        use_amp=False,
    )

    return model_config, train_config


@pytest.fixture
def smoke_output_dir(tmp_path: Path):
    """Temporary output directory for BLSTM FL smoke test."""

    output_dir = tmp_path / "smoke_test_bilstm_fl"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield output_dir
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)


def test_bilstm_fl_training_smoke(tiny_dataset, smoke_configs, smoke_output_dir):
    """Smoke test: run 1 epoch of BLSTM FL training on tiny feature dataset."""

    set_seed(42)
    device = torch.device("cpu")

    model_config, train_config = smoke_configs

    # Split dataset
    train_dataset, val_dataset, _ = split_dataset(tiny_dataset, seed=42)
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0

    # Validate labels are within the C1 classification set
    all_scores = tiny_dataset.data["c1"].to_numpy().copy()
    validate_scores_for_ce(torch.from_numpy(all_scores))

    # Data loaders
    train_loader = create_data_loader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_data_loader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Model
    model = BiLSTMClassifier(model_config)

    # Target scaler (kept for API parity)
    target_scaler = TargetScaler("none")
    target_scaler.fit(np.array(all_scores))

    # Class frequencies for alpha path in trainer (even though gamma-search typically overrides)
    class_freqs = get_class_frequencies(train_dataset)
    assert sum(class_freqs) > 0

    serialization_config = SerializationConfig(
        output_dir=smoke_output_dir, save_best_only=False, keep_last_k=1
    )

    trainer = BiLSTMFLTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_config=model_config,
        train_config=train_config,
        serialization_config=serialization_config,
        target_scaler=target_scaler,
        device=device,
        class_frequencies=class_freqs,
    )

    best_metrics = trainer.train()

    # Metrics sanity
    for key in ["mae", "rmse", "qwk", "kappa"]:
        assert key in best_metrics, f"Metric {key} missing"
        val = float(best_metrics[key])
        assert not np.isnan(val), f"{key} is NaN"
        assert not np.isinf(val), f"{key} is inf"

    # MAE should be on [0, 200]
    assert 0 <= best_metrics["mae"] <= 200

    # Checkpoint and metadata
    latest = smoke_output_dir / "latest.pt"
    assert latest.exists(), "Latest checkpoint not saved"
    ckpt = torch.load(latest, map_location="cpu")
    assert ckpt.get("loss_type") == "FocalLoss"
    assert "focal_gamma" in ckpt

    # Predictions validity
    if getattr(trainer, "best_val_predictions", None):
        preds = trainer.best_val_predictions
        valid_scores = {0, 40, 80, 120, 160, 200}
        for row in preds:
            assert row["pred"] in valid_scores
            assert row["pred_snapped"] in valid_scores

    logger.info("BLSTM FL smoke test passed with MAE=%.3f", best_metrics["mae"])