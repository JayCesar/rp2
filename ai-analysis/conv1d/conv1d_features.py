"""Train Conv1D model on grammar error features for C1 score prediction

Architecture based on Table 4 (Comp. 1) from CLaRiCe paper.
Uses kernel_size=1 for feature vectors since they don't have sequential structure.

Usage:
    uv run ai-analysis/conv1d/conv1d_features.py
"""

import json
import pathlib
import re
import sys

import polars as pl
import polars.selectors as cs
import torch

# Setup paths
script_path = pathlib.Path(__file__).resolve()
project_root = script_path.parent.parent.parent

# Add ai-analysis to path for imports
if str(project_root / "ai-analysis") not in sys.path:
    sys.path.insert(0, str(project_root / "ai-analysis"))

from feature_extraction.utils import logger
from blstm.blstm import MetricsAccumulator, TargetScaler, get_device, set_seed
from conv1d import (
    Conv1DRegressor,
    EssayDataset,
    ModelConfig,
    SerializationConfig,
    TrainConfig,
    Trainer,
    create_data_loader,
    split_dataset,
)


def main():
    """Main training workflow for Conv1D on grammar error features."""
    logger.info("=" * 70)
    logger.info("Conv1D Training on Grammar Error Features (LanguageTool)")
    logger.info("=" * 70)
    
    # Setup
    device = get_device("auto")
    set_seed(42)
    logger.info(f"Using device: {device}")
    
    # Enable performance optimizations for CUDA
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled CUDA performance optimizations")
    
    # Load dataset using pathlib - try different formats
    data_dir = project_root / "generated_datasets"
    data_filename = "dataset_with_languagetool_metrics"
    
    data_path = None
    for ext in ["parquet", "csv", "json"]:
        candidate = data_dir / f"{data_filename}.{ext}"
        if candidate.exists():
            data_path = candidate
            break
    
    if data_path is None:
        logger.error(f"Dataset not found: {data_dir / data_filename}.*")
        return 1
    
    logger.info(f"Loading dataset from {data_path}")
    
    # Load and select feature columns
    pattern = re.compile(r"^[A-Z0-9_]+$")
    feature_columns = [pl.col("c1"), cs.matches(r"^[A-Z0-9_]+$")]
    
    if data_path.suffix == ".parquet":
        df = (
            pl.scan_parquet(data_path)
            .select(feature_columns)
            .drop_nulls()
            .unique()
            .collect()
        )
    elif data_path.suffix == ".csv":
        df = (
            pl.scan_csv(data_path)
            .select(feature_columns)
            .drop_nulls()
            .unique()
            .collect()
        )
    else:  # json
        df = pl.read_json(data_path).select(feature_columns).drop_nulls().unique()
    
    # Get feature column names
    feature_cols = [c for c in df.columns if c != "c1" and pattern.match(c)]
    num_features = len(feature_cols)
    
    logger.info(f"Loaded {len(df)} essays with {num_features} features")
    logger.info(f"Features: {', '.join(feature_cols[:5])}...")
    
    # Split dataset
    logger.info("Splitting dataset (75/10/15)")
    train_df, val_df, test_df = split_dataset(df, val_ratio=0.10, test_ratio=0.15, seed=42)
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets
    train_dataset = EssayDataset(train_df)
    val_dataset = EssayDataset(val_df)
    test_dataset = EssayDataset(test_df)
    
    # Create data loaders
    num_workers = 4 if device.type == "cuda" else 2
    
    train_loader = create_data_loader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=num_workers, pin_memory=device.type == "cuda"
    )
    val_loader = create_data_loader(
        val_dataset, batch_size=32, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda"
    )
    test_loader = create_data_loader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda"
    )
    
    # Setup target scaler
    all_scores = df["c1"].to_numpy()
    target_scaler = TargetScaler("none")
    target_scaler.fit(all_scores)
    
    # Model configuration - use kernel_size=1 for features
    model_config = ModelConfig(
        input_dim=num_features,
        conv_filters=[28, 39],
        kernel_sizes=[1, 1],  # kernel_size=1 for feature vectors
        dense_neurons=90,
        dropout=0.303,
        pooling="max",
    )
    
    # Training configuration
    train_config = TrainConfig(
        epochs=50,
        batch_size=32,
        lr=7.06e-03,
        weight_decay=6.61e-04,
    )
    
    logger.info(f"Model params: {sum(p.numel() for p in Conv1DRegressor(model_config).parameters()):,}")
    
    # Setup output directory
    output_dir = script_path.parent / "runs" / "features"
    serialization_config = SerializationConfig(output_dir=output_dir)
    
    # Create and train model
    model = Conv1DRegressor(model_config)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_config=model_config,
        train_config=train_config,
        serialization_config=serialization_config,
        target_scaler=target_scaler,
        device=device,
    )
    
    logger.info("\nStarting training...")
    best_metrics = trainer.train()
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    model.eval()
    test_metrics = MetricsAccumulator()
    
    with torch.no_grad():
        for batch in test_loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True) if batch["lengths"] is not None else None
            targets = batch["targets"]
            ids = batch["ids"]
            
            if device.type == "cuda":
                with torch.autocast("cuda",enabled=True, dtype=torch.bfloat16):
                    predictions = model(tokens, lengths)
            else:
                predictions = model(tokens, lengths)
            
            predictions = torch.clamp(predictions, 0, 200)
            test_metrics.update(predictions.cpu(), targets, ids)
    
    test_computed = test_metrics.compute_metrics(target_scaler)
    
    logger.info("\nTest Results:")
    for metric, value in test_computed.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Save results
    logger.info("\nSaving results...")
    
    pl.DataFrame(best_metrics).write_csv(output_dir / "final_validation_metrics.csv")
    pl.DataFrame(best_metrics).write_parquet(output_dir / "final_validation_metrics.parquet")
    pl.DataFrame(test_computed).write_csv(output_dir / "final_test_metrics.csv")
    pl.DataFrame(test_computed).write_parquet(output_dir / "final_test_metrics.parquet")
    
    (output_dir / "model_config.json").write_text(json.dumps(model_config.to_dict(), indent=2))
    (output_dir / "train_config.json").write_text(json.dumps(train_config.to_dict(), indent=2))
    
    # Save training summary
    summary = []
    summary.append("Conv1D Training on Grammar Error Features\n" + "=" * 70 + "\n")
    summary.append(f"\nFeatures: {num_features} grammar error metrics\n")
    summary.append(f"Architecture: {model_config.to_dict()}\n")
    summary.append(f"Training: {train_config.to_dict()}\n")
    summary.append(f"\nDataset: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}\n")
    summary.append(f"\nBest Val Metrics: {json.dumps(best_metrics, indent=2)}\n")
    summary.append(f"Test Metrics: {json.dumps(test_computed, indent=2)}\n")
    summary.append("\nTraining History (last 10):\n")
    for entry in trainer.training_history[-10:]:
        summary.append(
            f"  Epoch {entry['epoch']}: Loss={entry['train_loss']:.4f}, "
            f"MAE={entry['val_mae']:.2f}, RMSE={entry['val_rmse']:.2f}, "
            f"R²={entry['r2']:.2f}, Time={entry['epoch_time']:.1f}s\n"
        )
    
    (output_dir / "training_summary.txt").write_text("".join(summary))
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Model: {output_dir}")
    logger.info(f"Val MAE: {best_metrics['mae']:.4f}, Test MAE: {test_computed['mae']:.4f}")
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"\nFailed: {e}", exc_info=True)
        exit(1)
