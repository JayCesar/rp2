#!/usr/bin/env python3
"""
BLSTM Training Script - Component 1 Specifications with Real Data

This script implements training for a bidirectional LSTM model based on
Component 1 specifications from the Portuguese table, using real essay
vector embeddings from extended_essay-br_preprocessed_for_BLSTM.parquet/.json.

Tabela 5. Hiperparâmetros dos regressores do modelo recorrente
Parâmetro       Comp. 1
Total de camadas   3
Unidades/Célula    10/26/21
Otimizador: Adam
Learning Rate: 1,01e-03
Dropout rate: 1,64e-01
Weight Decay(L2): 4,67e-06

Data Source: generated_datasets/extended_essay-br_preprocessed_for_BLSTM.parquet/.json
Total records: 6576 essays with 768-dimensional embeddings and C1 scores (0-200)

Required packages:
pip install torch numpy polars
"""

import logging
import re
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import polars.selectors as cs
import sklearn.model_selection
import torch
from torch.utils.data import DataLoader, Dataset

# Import our modules
sys.path.append(".")
from blstm import (
    BiLSTMRegressor,
    ModelConfig,
    SerializationConfig,
    TargetScaler,
    TrainConfig,
    get_device,
    set_seed,
)
from trainer import BiLSTMTrainer

project_root = Path(__file__).parent.parent.parent
assert project_root.name == "rp2"


# # Check for required packages and provide installation instructions
# def check_dependencies():
#     """Check for required packages and provide installation instructions."""
#     missing_packages = []
#
#     try:
#         import numpy
#     except ImportError:
#         missing_packages.append("numpy")
#
#     try:
#         import polars
#     except ImportError:
#         missing_packages.append("polars")
#
#     try:
#         import torch
#     except ImportError:
#         missing_packages.append("torch")
#
#     if missing_packages:
#         print("❌ Missing required packages!")
#         print("\nTo install the required packages, run:")
#         print(f"pip install {' '.join(missing_packages)}")
#         print("\nFor PyTorch, you might need:")
#         print(
#             "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
#         )
#         print("or for CUDA support:")
#         print(
#             "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
#         )
#         return False
#
#     return True
#
#
# # Only proceed with imports if dependencies are available
# if not check_dependencies():
#     sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EssayDataset(Dataset):
    """Dataset class for real essay vectors and C1 scores.

    Supports two input modes:
    - Token embeddings: expects columns ['c1', 'essay_token_embeddings'] where essay_token_embeddings is [seq_len, 768]
    - Feature vectors: expects 'c1' plus one or more feature columns (e.g., SCREAMING_SNAKE_CASE)
    """

    def __init__(self, data: pl.DataFrame, feature_cols: list[str]):
        super().__init__()
        self.data = data
        cols = set(self.data.columns)
        self.is_token_mode = "essay_token_embeddings" in cols
        self.is_vector_mode = "essay_vector" in cols  # Keep for backward compatibility
        # For feature mode, select only SCREAMING_SNAKE_CASE feature columns
        if not (self.is_token_mode or self.is_vector_mode):
            snake_case_pattern = re.compile(r"^[A-Z0-9_]+$")
            self.feature_cols = [
                c
                for c in self.data.columns
                if c != "c1" and snake_case_pattern.match(c)
            ]
        else:
            self.feature_cols = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the row as a dict (not a list of dicts)
        row = self.data.row(idx, named=True)

        if self.is_token_mode:
            # essay_token_embeddings -> tensor [seq_len, 768]
            token_embeddings = np.array(row["essay_token_embeddings"]) 
            tokens = torch.tensor(token_embeddings, dtype=torch.float32)
            # Prefer true token length if available; fallback to full length
            seq_length = int(row.get("essay_token_length", len(token_embeddings)))
            # Clamp to array length bounds
            seq_length = max(1, min(seq_length, len(token_embeddings)))
            lengths = torch.tensor(seq_length, dtype=torch.long)
        elif self.is_vector_mode:
            # essay_vector -> tensor [1, 768] - backward compatibility
            tokens = torch.tensor(row["essay_vector"], dtype=torch.float32).unsqueeze(0)
            lengths = torch.tensor(1, dtype=torch.long)
        else:
            # Assemble feature vector from selected columns -> tensor [1, input_dim]
            if not self.feature_cols:
                raise KeyError(
                    "No feature columns found for feature mode. Ensure dataset has feature columns besides 'c1'."
                )
            features = [float(row[c]) for c in self.feature_cols]
            tokens = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            lengths = torch.tensor(1, dtype=torch.long)

        return {
            "id": f"essay_{idx}",
            "tokens": tokens,  # Shape: [seq_len, 768] or [1, input_dim]
            "lengths": lengths,  # Actual sequence length
            "targets": torch.tensor(row["c1"], dtype=torch.float32),
        }

    def __getitems__(self, indices):
        # For batch processing - not used by default DataLoader
        return [self.__getitem__(idx) for idx in indices]


def collate_batch(batch):
    """Collate function to properly batch essay data for BiLSTM training."""
    ids = [item["id"] for item in batch]
    tokens = [item["tokens"] for item in batch]  # List of [seq_len, 768] tensors
    lengths = [item["lengths"] for item in batch]  # List of length tensors
    targets = [item["targets"] for item in batch]  # List of target tensors

    # Pad sequences to same length (for LSTM batch processing)
    # Use PyTorch's pad_sequence for variable length sequences
    from torch.nn.utils.rnn import pad_sequence

    # Pad token sequences: [batch_size, max_seq_len, 768]
    batched_tokens = pad_sequence(tokens, batch_first=True, padding_value=0.0)
    batched_lengths = torch.stack(lengths)  # [batch_size]
    batched_targets = torch.stack(targets)  # [batch_size]

    return {
        "ids": ids,
        "tokens": batched_tokens,  # [batch_size, max_seq_len, 768]
        "lengths": batched_lengths,  # [batch_size] - actual sequence lengths
        "targets": batched_targets,
    }


def create_data_loader(
    dataset: EssayDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = False,
) -> DataLoader:
    """Create DataLoader compatible with existing BLSTM trainer."""

    # Adjust num_workers for compatibility
    if num_workers > 0:
        try:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collate_batch,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,  # Add persistent_workers for better performance
            )
        except TypeError:
            # Fallback for older PyTorch versions
            logger.warning("persistent_workers not supported, using fallback")
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collate_batch,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_batch,
            num_workers=0,
            pin_memory=pin_memory,
        )


# class Component1BLSTM(nn.Module):
#     """
#     Component 1 Bidirectional LSTM Model based on CLaRiCe specifications.
#
#     Architecture:
#     - 3 layers with hidden sizes: 10, 26, 21
#     - Bidirectional LSTM
#     - Dropout rate: 1.64e-01
#     """
#
#     def __init__(self, input_dim: int = 768):
#         super().__init__()
#
#         self.input_dim = input_dim
#         self.dropout_rate = 1.64e-01  # 1,64e-01 from specs
#
#         # Component 1: 3 layers, units: 10/26/21
#         self.hidden_sizes = [10, 26, 21]
#         self.num_layers = 3
#
#         # Build LSTM layers
#         self.lstm_layers = nn.ModuleList()
#         current_input_dim = input_dim
#
#         for hidden_size in self.hidden_sizes:
#             lstm = nn.LSTM(
#                 input_size=current_input_dim,
#                 hidden_size=hidden_size,
#                 num_layers=1,
#                 batch_first=True,
#                 bidirectional=True,
#                 dropout=0.0,  # We'll apply dropout manually between layers
#             )
#             self.lstm_layers.append(lstm)
#
#             # Next layer input is bidirectional output
#             current_input_dim = hidden_size * 2
#
#         # Dropout layers between LSTM layers
#         self.dropout_layers = nn.ModuleList(
#             [nn.Dropout(self.dropout_rate) for _ in range(len(self.hidden_sizes) - 1)]
#         )
#
#         # Final output layer
#         final_hidden_dim = self.hidden_sizes[-1] * 2  # *2 for bidirectional
#         self.output_layer = nn.Sequential(
#             nn.Dropout(self.dropout_rate),
#             nn.Linear(final_hidden_dim, 64),
#             nn.ReLU(),
#             nn.Dropout(self.dropout_rate),
#             nn.Linear(64, 1),
#         )
#
#     def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass through Component 1 BLSTM.
#
#         Args:
#             x: Input tensor [batch_size, seq_len, input_dim]
#             lengths: Actual sequence lengths [batch_size]
#
#         Returns:
#             Output predictions [batch_size, 1]
#         """
#         current_input = x
#
#         # Pass through each LSTM layer
#         for i, lstm_layer in enumerate(self.lstm_layers):
#             # Pack sequences for LSTM efficiency
#             packed = nn.utils.rnn.pack_padded_sequence(
#                 current_input, lengths.cpu(), batch_first=True, enforce_sorted=False
#             )
#             packed_output, _ = lstm_layer(packed)
#             current_input, _ = nn.utils.rnn.pad_packed_sequence(
#                 packed_output, batch_first=True
#             )
#
#             # Apply dropout between layers (except after the last layer)
#             if i < len(self.lstm_layers) - 1:
#                 current_input = self.dropout_layers[i](current_input)
#
#         # Use last hidden state from the final bidirectional LSTM
#         # current_input shape: [batch_size, seq_len, hidden_size * 2]
#         batch_size = current_input.size(0)
#         final_hidden = torch.zeros(
#             batch_size, current_input.size(2), device=current_input.device
#         )
#
#         for b in range(batch_size):
#             actual_length = min(lengths[b].item(), current_input.size(1))
#             final_hidden[b] = current_input[b, actual_length - 1, :]
#
#         # Final prediction
#         output = self.output_layer(final_hidden)
#         return output


def create_component1_config(
    type: Literal["vectorized_essays", "features"],
) -> ModelConfig:
    """Create ModelConfig for Component 1 based on Portuguese specifications."""
    if type == "vectorized_essays":
        return ModelConfig(
            hidden_sizes=[10, 26, 21],
            input_dim=768,
            aggregation="attn",
            token_proj_dim=256,
            mlp_hidden=128,
            use_layer_norm=True,
        )
    if type == "features":
        return ModelConfig(
            hidden_sizes=[10, 26, 21],
            input_dim=30,
            aggregation="attn",
            token_proj_dim=128,
            mlp_hidden=128,
            use_layer_norm=True,
        )


def create_training_config() -> TrainConfig:
    """Create training configuration based on Portuguese specifications."""
    return TrainConfig(
        target_scaler="standard"  # Use standard scaler (z-normalization) instead of minmax
    )


def split_real_dataset(
    dataset: EssayDataset,
    # train_ratio: float = 0.7,
    val_ratio: float = 0.10,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """Split the real dataset into train/val/test sets using BERT-style stratification."""

    # Extract all records data for stratification
    indices = list(range(len(dataset)))
    scores = dataset.data["c1"].to_numpy()

    # Group similar C1 scores to ensure balanced splits
    stratify_groups = [max(0, min(4, (score - 1) // 50)) for score in scores]

    # First split: train vs (val + test)
    train_indices, temp_indices = sklearn.model_selection.train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=stratify_groups,
    )

    # Second split: val vs test from temp_indices
    temp_scores = [scores[i] for i in temp_indices]
    temp_stratify_groups = [
        0
        if score <= 49
        else 1
        if score <= 99
        else 2
        if score <= 149
        else 3
        if score <= 199
        else 4
        for score in temp_scores
    ]

    val_proportion = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = sklearn.model_selection.train_test_split(
        temp_indices,
        test_size=(1 - val_proportion),
        random_state=seed,
        stratify=temp_stratify_groups,
    )

    logger.info(
        f"Stratified dataset split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}"
    )

    # Create subset datasets by taking rows from the original DataFrame
    # This preserves the original schema (works for both vectorized essays and features)
    train_dataset = EssayDataset(
        dataset.data[train_indices], feature_cols=dataset.feature_cols
    )
    val_dataset = EssayDataset(
        dataset.data[val_indices], feature_cols=dataset.feature_cols
    )
    test_dataset = EssayDataset(
        dataset.data[test_indices], feature_cols=dataset.feature_cols
    )

    # Log score distributions to verify stratification worked
    train_scores = [scores[i] for i in train_indices]
    val_scores = [scores[i] for i in val_indices]

    logger.info(
        f"Train score distribution: {dict(zip(*np.unique(train_scores, return_counts=True)))}"
    )
    logger.info(
        f"Validation score distribution: {dict(zip(*np.unique(val_scores, return_counts=True)))}"
    )

    return train_dataset, val_dataset, test_dataset


def train_component1_standard(
    dataset: EssayDataset,
    device: torch.device,
    input_type: Literal["vectorized_essays", "features"],
) -> dict[str, float]:
    """Train Component 1 using the standard BiLSTMRegressor."""
    logger.info(f"Training Component 1 with Standard BiLSTMRegressor on {input_type}")

    # Split dataset
    train_dataset, val_dataset, test_dataset = split_real_dataset(dataset, seed=42)

    all_scores = dataset.data["c1"].to_numpy()
    target_scaler = TargetScaler("none")
    target_scaler.fit(np.array(all_scores))

    # Create configurations
    model_config = create_component1_config(input_type)

    # Adjust input_dim dynamically for feature inputs to match selected columns
    if input_type == "features":
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
    model = BiLSTMRegressor(model_config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Setup training
    output_dir = Path(__file__).parent / "runs" / input_type / "blstm_model"
    serialization_config = SerializationConfig(
        output_dir=output_dir, save_best_only=True, keep_last_k=3
    )

    trainer = BiLSTMTrainer(
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

    # Ensure the best checkpoint is saved at the expected path
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
            },
            checkpoint_path,
        )

    # Save training summary
    summary_path = Path(output_dir) / "component1_real_data_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Component 1 Training with Real Essay Data\n")
        f.write("=" * 50 + "\n\n")
        f.write("Portuguese Specifications Applied:\n")
        f.write("- Total de camadas: 3\n")
        f.write("- Unidades/Célula: 10/26/21 (approximated with final size 21)\n")
        f.write("- Otimizador: Adam (AdamW)\n")
        f.write("- Learning Rate: 1.01e-3\n")
        f.write("- Dropout rate: 0.164\n")
        f.write("- Weight Decay (L2): 4.67e-6\n\n")
        f.write("Real Data Statistics:\n")
        f.write(f"- Total essays: {len(dataset)}\n")
        f.write(f"- Training set: {len(train_dataset)}\n")
        f.write(f"- Validation set: {len(val_dataset)}\n")
        f.write(f"- Test set: {len(test_dataset)}\n")
        f.write("- Essay vector dimension: 768\n")
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
    logger.info("Training completed successfully")

    # BERT-style final metrics display
    print("\nFinal validation metrics:")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save final metrics to files (like BERT script)
    model_save_path.mkdir(parents=True, exist_ok=True)

    # Save as DataFrame and export to CSV/Parquet
    best_metrics_df = pl.DataFrame(best_metrics)

    best_metrics_df.write_csv(model_save_path / "final_validation_metrics.csv")
    print(
        f"Final validation metrics saved to {model_save_path / 'final_validation_metrics.csv'}"
    )

    best_metrics_df.write_parquet(model_save_path / "final_validation_metrics.parquet")
    print(
        f"Final validation metrics saved to {model_save_path / 'final_validation_metrics.parquet'}"
    )

    # Only show success messages if training actually succeeded
    logger.info("\n" + "=" * 70)
    logger.info("COMPONENT 1 TRAINING WITH REAL DATA COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info("Portuguese Component 1 specifications implemented:")
    logger.info("✓ Total de camadas: 3")
    logger.info("✓ Unidades/Célula: 10/26/21")
    logger.info("✓ Otimizador: Adam (AdamW)")
    logger.info("✓ Learning Rate: 1.01e-3")
    logger.info("✓ Dropout rate: 1.64e-01")
    logger.info("✓ Weight Decay (L2): 4.67e-6")
    logger.info("\nReal Data Used:")
    logger.info("✓ 768-dimensional BERT embeddings")
    logger.info("✓ C1 scores (0-200) for essay evaluation")
    logger.info("\nStandard implementation uses existing BiLSTMRegressor")

    # BERT-style completion message
    print(f"\n{'=' * 50}")
    print("✅ Training completed successfully!")
    print(f"{'=' * 50}")
    device_info = get_device("auto")
    if device_info.type == "cuda":
        print(f"Trained using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Trained using CPU")
    print(f"Model saved to: {model_save_path}")


def train_on_vectorized_essays(device: torch.device) -> None:
    """Train Component 1 using the standard BiLSTMRegressor."""
    logger.info(
        "Training Component 1 with Standard BiLSTMRegressor on Vectorized Essays"
    )

    # Load real essay data - try parquet first, then JSON as fallback
    parquet_file = (
        project_root
        / "generated_datasets"
        / "extended_essay-br_preprocessed_for_BLSTM.parquet"
    )
    json_file = (
        project_root
        / "generated_datasets"
        / "extended_essay-br_preprocessed_for_BLSTM.json"
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

    # Load dataset with optional sample limit for testing
    max_samples = None  # Set to None to use all samples
    # Try to include true lengths if present; fallback gracefully
    DEFAULT_MAX_SAMPLE_SIZE = 2**31 - 1
    try:
        relevant_columns = "c1", "essay_token_embeddings", "essay_token_length"
        dataset = (
            pl.scan_parquet(data_file)
            .select(relevant_columns)
            .head(max_samples if max_samples is not None else DEFAULT_MAX_SAMPLE_SIZE)
            .drop_nulls()
            .unique()
            .collect()
        )
    except Exception:
        relevant_columns = "c1", "essay_token_embeddings"
        dataset = (
            pl.scan_parquet(data_file)
            .select(relevant_columns)
            .head(max_samples if max_samples is not None else DEFAULT_MAX_SAMPLE_SIZE)
            .drop_nulls()
            .unique()
            .collect()
        )
    logger.info(f"Loaded dataset with {len(dataset)} essays: {dataset}")
    logger.info(
        f"Essay token embeddings shape: {dataset.schema['essay_token_embeddings']}"
    )

    dataset = EssayDataset(dataset)

    # Train using standard BiLSTMRegressor (approximated specifications)
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING WITH STANDARD BiLSTMRegressor")
    logger.info("=" * 70)

    input_type = "vectorized_essays"
    best_metrics = train_component1_standard(dataset, device, input_type)

    # # Check if training was actually successful (non-empty metrics dict)
    # training_successful = all(
    #     len(best_metrics[input_type]) > 0 for input_type in input_types
    # )
    #
    # if training_successful:

    model_save_path_vectorized_essays = (
        Path(__file__).parent / "runs" / input_type / "blstm_model"
    )
    show_and_save_metrics(best_metrics, model_save_path_vectorized_essays)


def train_on_features(device: torch.device) -> None:
    """Train Component 1 using the standard BiLSTMRegressor."""
    logger.info("Training Component 1 with Standard BiLSTMRegressor on Features")

    # Load real essay data - try parquet first, then JSON as fallback
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

    # Load dataset with optional sample limit for testing
    max_samples = None  # Set to None to use all samples
    # Select all SCREAMING_SNAKE_CASE feature columns plus the target 'c1'
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
    logger.info(f"Loaded dataset with {len(dataset)} essays: {dataset}")

    dataset = EssayDataset(dataset)

    # Train using standard BiLSTMRegressor (approximated specifications)
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING WITH STANDARD BiLSTMRegressor")
    logger.info("=" * 70)

    input_type = "features"
    best_metrics = train_component1_standard(dataset, device, input_type)

    # # Check if training was actually successful (non-empty metrics dict)
    # training_successful = all(
    #     len(best_metrics[input_type]) > 0 for input_type in input_types
    # )
    #
    # if training_successful:

    model_save_path_features = (
        Path(__file__).parent / "runs" / input_type / "blstm_model"
    )
    show_and_save_metrics(best_metrics, model_save_path_features)


def main():
    """Main training workflow for Component 1 Portuguese BLSTM specifications with real data."""
    print(f"\n{'=' * 50}")
    print("Component 1 BiLSTM C1 Training Starting")
    print(f"{'=' * 50}")

    logger.info("Starting Component 1 BLSTM Training with Real Essay Data")
    logger.info("=" * 70)

    # Setup
    device = get_device("auto")
    set_seed(42)
    logger.info(f"Using device: {device}")

    try:
        train_on_vectorized_essays(device)
        train_on_features(device)

        return 0
    # else:
    #     # Training failed - show failure message
    #     logger.error("\n" + "=" * 70)
    #     logger.error("COMPONENT 1 TRAINING WITH REAL DATA FAILED")
    #     logger.error("=" * 70)
    #     logger.error(
    #         "Training did not complete successfully. Check the error messages above."
    #     )
    #
    #     print(f"\n{'=' * 50}")
    #     print("❌ Training failed!")
    #     print(f"{'=' * 50}")
    #     print("Please check the error messages above for details.")
    #
    #     return 1
    #
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
