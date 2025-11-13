#!/usr/bin/env python3
"""Smoke tests for Conv1D CrossEntropy Loss training integration.

Validates end-to-end integration by running 1 epoch on a tiny dataset:
- Data loading and preprocessing
- Model instantiation
- Training loop with CE loss
- Validation with metric computation
- Checkpoint and prediction saving
- Predictions are valid scores from {0, 40, 80, 120, 160, 200}

Mirrors test_smoke_cross_entropy_loss.py but for Conv1D.
"""

import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
import pytest
import torch
from torch.utils.data import DataLoader

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "ai-analysis" / "conv1d"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ai-analysis"))

from conv1d import (
    ModelConfig,
    SerializationConfig,
    TrainConfig,
)
from conv1d_cross_entropy_loss import Conv1DClassifier, validate_scores_for_ce
from trainer_cross_entropy_loss import Conv1DCETrainer
from common import EssayDataset, TargetScaler, create_data_loader, split_dataset, set_seed

project_root = Path(__file__).parent.parent
logger = logging.getLogger(__name__)


@pytest.fixture
def tiny_dataset():
    """Load a tiny dataset for smoke testing."""
    parquet_file = (
        project_root / "generated_datasets" / "dataset_with_languagetool_metrics.parquet"
    )
    
    if not parquet_file.exists():
        pytest.skip(f"Dataset not found: {parquet_file}")
    
    # Load first 500 samples for stratification stability
    relevant_columns = [pl.col("c1"), cs.matches(r"^[A-Z0-9_]+$")]
    dataset = (
        pl.scan_parquet(parquet_file)
        .select(relevant_columns)
        .head(500)
        .drop_nulls()
        .unique()
        .collect()
    )
    
    return EssayDataset(dataset)


@pytest.fixture
def smoke_configs(tiny_dataset):
    """Create minimal configs for smoke testing."""
    # Model config - small Conv1D
    model_config = ModelConfig(
        conv_filters=[16, 32],  # Small filters for fast tests
        kernel_sizes=[3, 3],
        input_dim=len(tiny_dataset.feature_cols),
        dense_neurons=32,  # Small dense layer
        dropout=0.1,
        pooling="max",
    )
    
    # Train config - 1 epoch, fast
    train_config = TrainConfig(
        epochs=1,
        batch_size=16,
        lr=1e-3,
        optimizer="adamw",
        early_stopping_patience=999,  # Don't stop
        use_amp=False,  # Simpler for testing
    )
    
    return model_config, train_config


@pytest.fixture
def smoke_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "smoke_test_conv1d_ce"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
    # Cleanup after test
    if output_dir.exists():
        shutil.rmtree(output_dir)


def test_conv1d_ce_training_smoke(tiny_dataset, smoke_configs, smoke_output_dir):
    """Smoke test: Run 1 epoch of Conv1D CE training on tiny dataset."""
    set_seed(42)
    device = torch.device("cpu")  # Use CPU for reproducibility
    
    model_config, train_config = smoke_configs
    
    # Split dataset
    train_dataset, val_dataset, _ = split_dataset(tiny_dataset, seed=42)
    assert len(train_dataset) > 0, "Train dataset empty"
    assert len(val_dataset) > 0, "Val dataset empty"
    
    # Verify all targets are valid CE scores
    all_scores = tiny_dataset.data["c1"].to_numpy().copy()
    validate_scores_for_ce(torch.from_numpy(all_scores))
    
    # Create loaders
    train_loader = create_data_loader(
        train_dataset, batch_size=train_config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = create_data_loader(
        val_dataset, batch_size=train_config.batch_size, shuffle=False, num_workers=0
    )
    
    # Create model
    model = Conv1DClassifier(model_config)
    
    # Create trainer
    target_scaler = TargetScaler("none")
    target_scaler.fit(np.array(all_scores))
    
    serialization_config = SerializationConfig(
        output_dir=smoke_output_dir, save_best_only=False, keep_last_k=1
    )
    
    trainer = Conv1DCETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_config=model_config,
        train_config=train_config,
        serialization_config=serialization_config,
        target_scaler=target_scaler,
        device=device,
    )
    
    # Train 1 epoch
    best_metrics = trainer.train()
    
    # Verify metrics were computed
    assert "mae" in best_metrics, "MAE metric missing"
    assert "rmse" in best_metrics, "RMSE metric missing"
    assert "qwk" in best_metrics, "QWK metric missing"
    assert "kappa" in best_metrics, "Kappa metric missing"
    
    # Verify metrics are not NaN
    for metric_name, metric_value in best_metrics.items():
        assert not np.isnan(metric_value), f"{metric_name} is NaN"
        assert not np.isinf(metric_value), f"{metric_name} is inf"
    
    # Verify MAE is reasonable (0-200 scale)
    assert 0 <= best_metrics["mae"] <= 200, f"MAE out of range: {best_metrics['mae']}"
    
    # Verify checkpoints were saved
    latest_checkpoint = smoke_output_dir / "latest.pt"
    assert latest_checkpoint.exists(), "Latest checkpoint not saved"
    
    # Verify checkpoint contains loss_type metadata
    checkpoint = torch.load(latest_checkpoint, map_location="cpu")
    assert "loss_type" in checkpoint, "loss_type missing from checkpoint"
    assert checkpoint["loss_type"] == "CrossEntropyLoss", "Wrong loss_type in checkpoint"
    
    # Verify predictions are valid scores
    if hasattr(trainer, "best_val_predictions") and trainer.best_val_predictions:
        predictions_list = trainer.best_val_predictions
        pred_scores = [p["pred"] for p in predictions_list]
        
        # All predictions should be from valid class set
        valid_scores = {0, 40, 80, 120, 160, 200}
        for pred in pred_scores:
            assert pred in valid_scores, f"Invalid prediction: {pred}"
    
    logger.info(f"Smoke test passed! Best MAE: {best_metrics['mae']:.2f}")


def test_conv1d_ce_features_script_imports():
    """Test that Conv1D CE training scripts can be imported without errors."""
    try:
        from conv1d_train_on_features_cross_entropy_loss import (
            create_component1_config_for_features,
            create_training_config,
        )
        from conv1d_train_on_vectorized_essays_cross_entropy_loss import (
            create_component1_config_for_vectorized_essays,
        )
        
        # Verify functions return valid configs
        features_model_config = create_component1_config_for_features()
        assert isinstance(features_model_config, ModelConfig)
        
        features_train_config = create_training_config()
        assert isinstance(features_train_config, TrainConfig)
        
        essays_model_config = create_component1_config_for_vectorized_essays()
        assert isinstance(essays_model_config, ModelConfig)
        
    except ImportError as e:
        pytest.fail(f"Failed to import Conv1D CE training scripts: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
