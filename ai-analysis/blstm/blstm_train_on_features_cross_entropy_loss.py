#!/usr/bin/env python3
"""BiLSTM Training on Features with CrossEntropy Loss

Trains a BiLSTMClassifier on essay linguistic features using CrossEntropyLoss
for 6-class C1 score classification {0, 40, 80, 120, 160, 200}.

Mirrors blstm_train_on_features.py but uses classification approach:
- BiLSTMClassifier with 6 logits output
- CrossEntropyLoss on class indices
- Predictions converted back to score space for metrics
- Output directory: runs/features_cross_entropy_loss/

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
from torch.utils.data import DataLoader

# Import modules
sys.path.append(".")
from blstm import (
    ModelConfig,
    SerializationConfig,
    TargetScaler,
    TrainConfig,
    get_device,
    set_seed,
)

# Import CE-specific components
from blstm_cross_entropy_loss import BiLSTMClassifier
from trainer_cross_entropy_loss import BiLSTMCETrainer

# Import common modules
sys.path.append(str(Path(__file__).parent.parent))
from common import (
    EssayDataset,
    create_data_loader,
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
        hidden_sizes=[10, 26, 21],
        input_dim=30,  # Will be adjusted dynamically
        aggregation="attn",
        token_proj_dim=128,
        mlp_hidden=128,
        use_layer_norm=True,
    )


def create_training_config() -> TrainConfig:
    """Create training configuration."""
    return TrainConfig(
        target_scaler="standard"  # Use standard scaler (z-normalization)
    )


def train_on_features_ce(device: torch.device) -> dict[str, float]:
    """Train Component 1 using BiLSTMClassifier with CrossEntropyLoss on features."""
    logger.info("Training Component 1 with CE on Features")

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

    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, seed=42)

    # Target scaler (not used for CE but keep for API parity)
    all_scores = dataset.data["c1"].to_numpy()
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
    model = BiLSTMClassifier(model_config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Setup training
    output_dir = Path(__file__).parent / "runs" / "features_cross_entropy_loss" / "blstm_model"
    serialization_config = SerializationConfig(
        output_dir=output_dir, save_best_only=True, keep_last_k=3
    )

    trainer = BiLSTMCETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_config=model_config,
        train_config=train_config,
        serialization_config=serialization_config,
        target_scaler=target_scaler,
        device=device,
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
        f.write("Component 1 Training on Features with CrossEntropyLoss\n")
        f.write("=" * 50 + "\n\n")
        f.write("Classification Specifications:\n")
        f.write("- Loss: CrossEntropyLoss\n")
        f.write("- Classes: 6 (0, 40, 80, 120, 160, 200)\n")
        f.write("- Model: BiLSTMClassifier\n")
        f.write("- Total de camadas: 3\n")
        f.write("- Unidades/Célula: 10/26/21\n")
        f.write("- Otimizador: AdamW\n")
        f.write("- Learning Rate: 1.01e-3\n")
        f.write("- Dropout rate: 0.164\n")
        f.write("- Weight Decay (L2): 4.67e-6\n\n")
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
    logger.info("COMPONENT 1 CE TRAINING ON FEATURES COMPLETED")
    logger.info("=" * 70)
    logger.info("Specifications implemented:")
    logger.info("✓ CrossEntropyLoss for 6-class classification")
    logger.info("✓ BiLSTMClassifier with attention aggregation")
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
    """Main training workflow for Component 1 CE on features."""
    print(f"\n{'=' * 50}")
    print("Component 1 BiLSTM CE Training on Features")
    print(f"{'=' * 50}")

    logger.info("Starting Component 1 CE Training on Features")
    logger.info("=" * 70)

    # Setup
    device = get_device("auto")
    set_seed(42)
    logger.info(f"Using device: {device}")

    try:
        best_metrics = train_on_features_ce(device)

        model_save_path = (
            Path(__file__).parent / "runs" / "features_cross_entropy_loss" / "blstm_model"
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
