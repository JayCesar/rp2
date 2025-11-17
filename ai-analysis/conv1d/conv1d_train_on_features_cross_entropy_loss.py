#!/usr/bin/env python3
"""Conv1D Training on Features with CrossEntropy Loss

Trains a Conv1DClassifier on essay linguistic features using CrossEntropyLoss
for 6-class C1 score classification {0, 40, 80, 120, 160, 200}.

Mirrors blstm_train_on_features_cross_entropy_loss.py but uses Conv1D:
- Conv1DClassifier with 6 logits output
- CrossEntropyLoss on class indices
- Predictions converted back to score space for metrics
- Output directory: runs/features_cross_entropy_loss/conv1d_model/

Data Source: dataset_with_languagetool_metrics.parquet
Features: SCREAMING_SNAKE_CASE linguistic feature columns
Target: C1 scores (0-200)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
import torch

# Import Conv1D modules
sys.path.append(".")
from conv1d import (
    ModelConfig,
    SerializationConfig,
    TrainConfig,
)

# Import CE-specific components
from conv1d_cross_entropy_loss import Conv1DClassifier, validate_scores_for_ce
from conv1d_trainer_cross_entropy_loss import Conv1DCETrainer
from calculate_class_frequencies import (
    get_class_frequencies,
    print_class_distribution,
)

# Import common modules
sys.path.append(str(Path(__file__).parent.parent))
from common import (
    EssayDataset,
    TargetScaler,
    create_data_loader,
    get_device,
    set_seed,
    split_dataset,
)

project_root = Path(__file__).parent.parent.parent
assert project_root.name == "rp2"

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_component1_config_for_features() -> ModelConfig:
    """Create ModelConfig for Component 1 features with CE."""
    return ModelConfig(
        conv_filters=[28, 39],  # From CLaRiCe Table 4
        kernel_sizes=[3, 3],
        input_dim=30,  # Adjusted dynamically
        dense_neurons=90,  # From CLaRiCe Table 4
        dropout=0.303,  # From CLaRiCe Table 4
        pooling="max",
    )


def create_training_config() -> TrainConfig:
    """Create training configuration."""
    return TrainConfig()  # Uses defaults from conv1d.py


def train_on_features_ce(device: torch.device) -> dict[str, float]:
    """Train Component 1 using Conv1DClassifier with CrossEntropyLoss on features."""
    logger.info("Training Component 1 with CE on Features (Conv1D)")

    # Load feature data
    parquet_file = (
        project_root
        / "generated_datasets"
        / "dataset_with_languagetool_metrics.parquet"
    )
    json_file = (
        project_root / "generated_datasets" / "dataset_with_languagetool_metrics.json"
    )

    data_file = None
    if Path(parquet_file).exists():
        data_file = parquet_file
        logger.info(f"Using parquet file: {data_file}")
    elif Path(json_file).exists():
        data_file = json_file
        logger.info(f"Using JSON file: {data_file}")
    else:
        logger.error(f"Data file not found. Looked for: {parquet_file} and {json_file}")
        exit(1)

    # Load dataset - select SCREAMING_SNAKE_CASE feature columns plus target 'c1'
    max_samples = None
    relevant_columns = [pl.col("c1"), cs.matches(r"^[A-Z0-9_]+$")]
    DEFAULT_MAX_SAMPLE_SIZE = 2**31 - 1
    dataset = (
        pl.scan_parquet(data_file)
        .select(relevant_columns)
        .head(max_samples if max_samples is not None else DEFAULT_MAX_SAMPLE_SIZE)
        .drop_nulls()
        .unique()
        .collect()
    )
    logger.info(f"Loaded dataset with {len(dataset)} essays")

    dataset = EssayDataset(dataset)

    # Validate strict label set
    all_scores = dataset.data["c1"].to_numpy()
    validate_scores_for_ce(torch.tensor(all_scores))
    logger.info("✓ All labels valid for CE training")

    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, seed=42)

    # Calculate class frequencies for inverse frequency weighting
    class_frequencies = get_class_frequencies(train_dataset)
    logger.info("Training set class distribution:")
    print_class_distribution(class_frequencies)

    # Target scaler (not used for CE but keep for API parity)
    target_scaler = TargetScaler("none")
    target_scaler.fit(np.array(all_scores))

    # Create configurations
    model_config = create_component1_config_for_features()

    # Adjust input_dim dynamically to match selected feature columns
    try:
        model_config.input_dim = len(dataset.feature_cols)
    except AttributeError:
        # Fallback: infer from first row length (excluding target)
        first_row = dataset.data.row(0, named=True)
        model_config.input_dim = len([k for k in first_row.keys() if k != "c1"])

    train_config = create_training_config()

    logger.info(f"Model Config: {model_config.to_dict()}")
    logger.info(f"Train Config: {train_config.to_dict()}")

    # Create data loaders
    train_loader = create_data_loader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    val_loader = create_data_loader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    # Create model
    model = Conv1DClassifier(model_config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Setup training
    output_dir = (
        Path(__file__).parent / "runs" / "features_cross_entropy_loss" / "conv1d_model"
    )
    serialization_config = SerializationConfig(
        output_dir=output_dir, save_best_only=True, keep_last_k=3
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
        class_frequencies=class_frequencies,
    )

    # Train model
    best_metrics = trainer.train()
    logger.info(f"Best metrics: {best_metrics}")

    # Save per-sample validation predictions for best epoch
    try:
        if getattr(trainer, "best_val_predictions", None):
            preds_df = pl.DataFrame(trainer.best_val_predictions)
            preds_dir = Path(output_dir)
            preds_dir.mkdir(parents=True, exist_ok=True)
            preds_csv = preds_dir / "validation_predictions_best.csv"
            preds_parquet = preds_dir / "validation_predictions_best.parquet"
            preds_df.write_csv(preds_csv)
            preds_df.write_parquet(preds_parquet)
            logger.info(
                f"Saved best validation predictions to: {preds_csv} and {preds_parquet}"
            )
        else:
            logger.warning("No best validation predictions captured to save.")
    except Exception as e:
        logger.warning(f"Failed to save best validation predictions: {e}")

    # Save checkpoint path
    checkpoint_path = Path(output_dir) / "best.pt"

    # Ensure the best checkpoint is saved
    try:
        trainer._save_checkpoint(is_best=True, metrics=best_metrics)
        logger.info(f"Best checkpoint saved to: {checkpoint_path}")
    except Exception as e:
        logger.warning(
            f"Failed to save checkpoint via trainer; saving minimal state to {checkpoint_path}: {e}"
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model_config.to_dict(),
                "train_config": train_config.to_dict(),
                "metrics": best_metrics,
                "loss_type": "CrossEntropyLoss",
            },
            checkpoint_path,
        )

    # Save training summary
    summary_path = Path(output_dir) / "component1_features_ce_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Component 1 Training on Features with CrossEntropyLoss (Conv1D)\n")
        f.write("=" * 50 + "\n\n")
        f.write("Classification Specifications:\n")
        f.write("- Loss: CrossEntropyLoss\n")
        f.write("- Classes: 6 (0, 40, 80, 120, 160, 200)\n")
        f.write("- Model: Conv1DClassifier\n")
        f.write("- Conv filters: 28, 39\n")
        f.write("- Dense neurons: 90\n")
        f.write("- Optimizer: AdamW\n")
        f.write("- Learning Rate: 7.06e-3\n")
        f.write("- Dropout rate: 0.303\n")
        f.write("- Weight Decay (L2): 6.61e-4\n\n")
        f.write("Data Statistics:\n")
        f.write(f"- Total essays: {len(dataset)}\n")
        f.write(f"- Training set: {len(train_dataset)}\n")
        f.write(f"- Validation set: {len(val_dataset)}\n")
        f.write(f"- Test set: {len(test_dataset)}\n")
        f.write(f"- Feature dimension: {model_config.input_dim}\n")
        f.write("- C1 score range: 0-200\n\n")
        f.write(f"Model Configuration:\n{model_config.to_dict()}\n\n")
        f.write(f"Training Configuration:\n{train_config.to_dict()}\n\n")
        f.write(f"Best Validation Metrics:\n{best_metrics}\n\n")
        f.write("Training History (last 10 epochs):\n")
        for entry in trainer.training_history[-10:]:
            f.write(
                f"  Epoch {entry['epoch']}: "
                f"Train Loss={entry['train_loss']:.6f}, "
                f"Val MAE={entry['val_mae']:.2f}, "
                f"Val RMSE={entry['val_rmse']:.2f}, "
                f"Val Kappa={entry['kappa']:.2f}, "
                f"Val QWK={entry['qwk']:.2f}, "
                f"Val R²={entry['r2']:.2f}, "
                f"Val Pearson={entry['pearson_corr']:.2f}, "
                f"Val Step Acc={entry['step_accuracy']:.3f}, "
                f"Time={entry['epoch_time']:.1f}s\n"
            )

    logger.info(f"Training summary saved to: {summary_path}")
    return best_metrics


def show_and_save_metrics(
    best_metrics: dict[str, float], model_save_path: Path
) -> None:
    """Show and save metrics."""
    logger.info("CE Training completed successfully")

    # Final metrics display
    print("\nFinal validation metrics (CrossEntropyLoss):")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save final metrics
    model_save_path.mkdir(parents=True, exist_ok=True)

    best_metrics_df = pl.DataFrame(best_metrics)

    best_metrics_df.write_csv(model_save_path / "final_validation_metrics.csv")
    print(
        f"Final validation metrics saved to {model_save_path / 'final_validation_metrics.csv'}"
    )

    best_metrics_df.write_parquet(model_save_path / "final_validation_metrics.parquet")
    print(
        f"Final validation metrics saved to {model_save_path / 'final_validation_metrics.parquet'}"
    )

    logger.info("\n" + "=" * 70)
    logger.info("COMPONENT 1 CE TRAINING ON FEATURES COMPLETED (CONV1D)")
    logger.info("=" * 70)
    logger.info("Specifications implemented:")
    logger.info("✓ CrossEntropyLoss for 6-class classification")
    logger.info("✓ Conv1DClassifier with max pooling")
    logger.info("✓ Classes: {0, 40, 80, 120, 160, 200}")
    logger.info("✓ All metrics computed in score space")

    print(f"\n{'=' * 50}")
    print("✅ CE Training completed successfully!")
    print(f"{'=' * 50}")
    device_info = get_device("auto")
    if device_info.type == "cuda":
        print(f"Trained using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Trained using CPU")
    print(f"Model saved to: {model_save_path}")


def main():
    """Main training workflow for Component 1 CE on features (Conv1D)."""
    print(f"\n{'=' * 50}")
    print("Component 1 Conv1D CE Training on Features")
    print(f"{'=' * 50}")

    logger.info("Starting Component 1 CE Training on Features (Conv1D)")
    logger.info("=" * 70)

    # Setup
    device = get_device("auto")
    set_seed(42)
    logger.info(f"Using device: {device}")

    try:
        best_metrics = train_on_features_ce(device)

        model_save_path = (
            Path(__file__).parent
            / "runs"
            / "features_cross_entropy_loss"
            / "conv1d_model"
        )
        show_and_save_metrics(best_metrics, model_save_path)

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
