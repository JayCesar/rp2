"""Bidirectional LSTM for Essay C1 Score Prediction

This module implements a flexible bidirectional LSTM model for predicting essay 
C1 scores (range 0-200 with steps of 40) from 768-dimensional BERT token embeddings.

Data Format Expectations:
- Input: CSV with columns for essay ID, path to embedding file (.npy or .pt), and C1 score
- Embedding files: torch tensors or numpy arrays with shape [sequence_length, 768]
- Target scores: integers in range 0-200, typically {0, 40, 80, 120, 160, 200}

Example Usage:
    # Training
    python bidirectional_lstm.py train --train-csv data.csv --val-split 0.2 \
        --hidden-size 256 --epochs 20 --batch-size 32
    
    # Evaluation
    python bidirectional_lstm.py eval --checkpoint runs/bilstm/best.pt \
        --test-csv test_data.csv
    
    # Prediction
    python bidirectional_lstm.py predict --checkpoint runs/bilstm/best.pt \
        --input-csv new_data.csv --output predictions.csv --snap-to-step

Model Architecture:
- Bidirectional LSTM processes token sequences
- Configurable aggregation: last hidden state, mean/max pooling, or attention
- MLP head for regression with optional layer normalization
- Supports mixed precision training and various optimizers

Metrics:
- Standard regression metrics (MAE, RMSE, R²)
- Step-aligned accuracy (after snapping to nearest 40-point increment)
- All metrics computed on original 0-200 scale
"""

import logging
import os
import pathlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import scipy.stats
import sklearn.metrics
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset, random_split


# Constants
class ScoreConstants:
    MIN = 0
    MAX = 200
    STEP = 40


# # Configure logging
# def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> None:
#     """Set up logging with console and optional file output."""
#     handlers = [logging.StreamHandler()]
#     if log_file:
#         os.makedirs(os.path.dirname(log_file), exist_ok=True)
#         handlers.append(logging.FileHandler(log_file))
#
#     logging.basicConfig(
#         level=level,
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         handlers=handlers,
#     )
#
#     # Reduce noise from some libraries
#     logging.getLogger("transformers").setLevel(logging.WARNING)
#     logging.getLogger("torch").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# Configuration dataclasses
@dataclass
class ModelConfig:
    """Configuration for the BiLSTM model architecture."""

    hidden_sizes: list[int]
    input_dim: int = 768
    num_layers: int = 3
    # bidirectional: bool = True
    dropout: float = 1.64e-01
    aggregation: Literal["last", "mean", "max", "attn"] = "last"
    mlp_hidden: int | None = None
    use_layer_norm: bool = False
    output_range: tuple[int, int] = (ScoreConstants.MIN, ScoreConstants.MAX)

    def to_dict(self) -> dict[str, int | float | bool | str | tuple[int, int]]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, d: dict[str, int | float | bool | str | tuple[int, int]]
    ) -> "ModelConfig":
        return cls(**d)


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    train_csv: str | None = None
    val_csv: str | None = None
    test_csv: str | None = None
    id_column: str = "id"
    embedding_column: str = "embedding_path"
    score_column: str = "c1"
    embedding_format: Literal["npy", "pt", "auto"] = "auto"
    max_seq_len: int = 1024
    pad_value: float = 0.0
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    val_split: float = 0.1
    test_split: float = 0.0
    shuffle: bool = True

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, str | int | float | bool | None]) -> "DataConfig":
        return cls(**d)


@dataclass
class TrainConfig:
    """Configuration for training parameters."""

    epochs: int = 25
    batch_size: int = 32
    lr: float = 1.01e-03
    weight_decay: float = 4.67e-06
    optimizer: Literal["adam", "adamw"] = "adamw"
    scheduler: Literal["plateau", "onecycle", "none"] = "none"
    plateau_patience: int = 2
    plateau_factor: float = 0.5
    onecycle_pct_start: float = 0.1
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 3
    seed: int = 42
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.bfloat16
    # compile: bool = False
    target_scaler: Literal["none", "minmax", "standard"] = "none"

    def to_dict(self) -> dict[str, str | int | float | bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, str | int | float | bool]) -> "TrainConfig":
        return cls(**d)


@dataclass
class SerializationConfig:
    """Configuration for model checkpointing and saving."""

    output_dir: pathlib.Path = pathlib.Path("runs/bilstm")
    save_best_only: bool = True
    keep_last_k: int = 3

    def to_dict(self) -> dict[str, str | int | bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, str | int | bool]) -> "SerializationConfig":
        return cls(**d)


# Custom Exceptions
class DataFormatError(Exception):
    """Raised when data format is invalid or incompatible."""

    pass


class ModelConfigError(Exception):
    """Raised when model configuration is invalid."""

    pass


# Utility functions
def get_device(preference: str = "auto") -> torch.device:
    """Auto-detect or select the best available device."""
    if preference != "auto":
        return torch.device(preference)

    # Check for CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        return device

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
        return device

    # Fallback to CPU
    device = torch.device("cpu")
    logger.info("Using CPU")
    return device


def set_seed(seed: int) -> None:
    """Set seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def snap_to_step(score: float, step: int = ScoreConstants.STEP) -> int:
    """Snap score to nearest step increment for evaluation."""
    return int(round(score / step) * step)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Comprehensive Metrics Functions (matching BERT script)
def quadratic_weighted_kappa(y_true, y_pred, labels=None):
    """Calculate Quadratic Weighted Kappa (QWK) score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of possible labels (optional)

    Returns:
        QWK score between -1 and 1, where 1 is perfect agreement
    """
    if labels is None:
        labels = sorted(list(set(y_true + y_pred)))

    # Create confusion matrix
    n_labels = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    confusion_matrix = np.zeros((n_labels, n_labels))
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = label_to_idx[true_label]
        pred_idx = label_to_idx[pred_label]
        confusion_matrix[true_idx, pred_idx] += 1

    # Normalize to get observed agreement matrix
    total = confusion_matrix.sum()
    if total == 0:
        return 0.0

    observed_matrix = confusion_matrix / total

    # Calculate expected agreement matrix
    row_marginals = confusion_matrix.sum(axis=1) / total
    col_marginals = confusion_matrix.sum(axis=0) / total
    expected_matrix = np.outer(row_marginals, col_marginals)

    # Create quadratic weight matrix
    weights = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            weights[i, j] = (i - j) ** 2 / (n_labels - 1) ** 2

    # Calculate weighted agreements
    observed_agreement = np.sum(weights * observed_matrix)
    expected_agreement = np.sum(weights * expected_matrix)

    # Calculate QWK
    if expected_agreement == 0:
        return 0.0

    qwk = 1 - (observed_agreement / expected_agreement)
    return qwk


def round_to_c1_levels(scores):
    """Round scores to nearest valid C1 levels (0, 40, 80, 120, 160, 200)"""
    c1_levels = [0, 40, 80, 120, 160, 200]
    rounded = []
    for score in scores:
        # Clamp to valid range first
        score = max(0, min(200, score))
        # Find closest C1 level
        closest_level = min(c1_levels, key=lambda x: abs(x - score))
        rounded.append(closest_level)
    return rounded


# Target scaling
class TargetScaler:
    """Simple target scaler with different modes."""

    def __init__(self, mode: Literal["none", "minmax", "standard"] = "minmax"):
        self.mode = mode
        self.fitted = False
        self.min_val = None
        self.max_val = None
        self.mean_val = None
        self.std_val = None

    def fit(self, targets: np.ndarray) -> "TargetScaler":
        """Fit the scaler to target values."""
        if self.mode == "minmax":
            self.min_val = targets.min()
            self.max_val = targets.max()
        elif self.mode == "standard":
            self.mean_val = targets.mean()
            self.std_val = targets.std()

        self.fitted = True
        return self

    def transform(self, targets: np.ndarray) -> np.ndarray:
        """Transform targets using fitted scaler."""
        if not self.fitted and self.mode != "none":
            raise ValueError("Scaler must be fitted before transform")

        if self.mode == "none":
            return targets
        elif self.mode == "minmax":
            return (targets - self.min_val) / (self.max_val - self.min_val + 1e-8)
        elif self.mode == "standard":
            return (targets - self.mean_val) / (self.std_val + 1e-8)

        return targets

    def inverse_transform(self, targets: np.ndarray) -> np.ndarray:
        """Inverse transform targets to original scale."""
        if not self.fitted and self.mode != "none":
            raise ValueError("Scaler must be fitted before inverse_transform")

        if self.mode == "none":
            return targets
        elif self.mode == "minmax":
            return targets * (self.max_val - self.min_val) + self.min_val
        elif self.mode == "standard":
            return targets * self.std_val + self.mean_val

        return targets

    def state_dict(self) -> dict[str, any]:
        """Get state dictionary for serialization."""
        return {
            "mode": self.mode,
            "fitted": self.fitted,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "mean_val": self.mean_val,
            "std_val": self.std_val,
        }

    def load_state_dict(self, state: dict[str, any]) -> None:
        """Load state from dictionary."""
        self.mode = state["mode"]
        self.fitted = state["fitted"]
        self.min_val = state.get("min_val")
        self.max_val = state.get("max_val")
        self.mean_val = state.get("mean_val")
        self.std_val = state.get("std_val")


class EssayDataset(Dataset):
    """Dataset class for real essay vectors and C1 scores."""

    def __init__(self, data: pl.DataFrame):
        super().__init__()
        self.records = data

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records.row(idx, named=True)

    def __getitems__(self, indices):
        return (
            self.records.with_row_index(name="index")
            .filter(pl.col("index").is_in(indices))
            .to_dicts()
        )


def collate_batch(
    batch: list[dict[str, any]], pad_value: float = 0.0
) -> dict[str, any]:
    """Collate function for variable-length sequences."""
    ids = [item["id"] for item in batch]
    embeddings = [item["embedding"] for item in batch]
    scores = [item["score"] for item in batch]

    # Get sequence lengths
    lengths = torch.tensor([emb.shape[0] for emb in embeddings], dtype=torch.long)

    # Pad sequences to same length
    padded_embeddings = pad_sequence(
        embeddings, batch_first=True, padding_value=pad_value
    )

    return {
        "tokens": padded_embeddings,
        "lengths": lengths,
        "targets": torch.tensor(scores, dtype=torch.float),
        "ids": ids,
    }


def split_dataset(
    dataset: EssayDataset,
    val_ratio: float = 0.1,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Split dataset into train/val/test."""
    total_size = len(dataset)
    val_size = int(val_ratio * total_size)
    test_size = int(test_ratio * total_size)
    train_size = total_size - val_size - test_size

    torch.manual_seed(seed)
    return random_split(dataset, [train_size, val_size, test_size])


# Model Components
class AttentionAggregation(nn.Module):
    """Attention-based sequence aggregation."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Apply attention aggregation."""
        # sequences: [batch_size, seq_len, hidden_size]
        # lengths: [batch_size]

        batch_size, seq_len, hidden_size = sequences.shape

        # Compute attention weights
        attention_weights = self.attention(sequences)  # [batch_size, seq_len, 1]

        # Create mask for padding
        mask = torch.arange(seq_len, device=sequences.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]

        # Apply mask to attention weights
        attention_weights = attention_weights.masked_fill(~mask, float("-inf"))
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Weighted sum
        aggregated = (sequences * attention_weights).sum(
            dim=1
        )  # [batch_size, hidden_size]

        return aggregated


class BiLSTMRegressor(nn.Module):
    """Bidirectional LSTM for essay C1 score regression."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # LSTM layers with per-layer hidden sizes
        hs1, hs2, hs3 = self.config.hidden_sizes[:3]
        direction_multiplier = 2  # bidirectional

        self.lstm1 = nn.LSTM(
            input_size=self.config.input_dim,
            hidden_size=hs1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=hs1 * direction_multiplier,
            hidden_size=hs2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm3 = nn.LSTM(
            input_size=hs2 * direction_multiplier,
            hidden_size=hs3,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Dropout between LSTM layers (num_layers=1 in each LSTM, so internal dropout is ignored)
        self.dropout = nn.Dropout(self.config.dropout)

        lstm_output_size = hs3 * direction_multiplier

        # Aggregation layer
        if config.aggregation == "attn":
            self.aggregation = AttentionAggregation(lstm_output_size)
        else:
            self.aggregation = None

        # Output head: single linear readout (no hidden MLP)
        self.head = nn.Linear(lstm_output_size, 1)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        batch_size = tokens.shape[0]

        # Pack sequences for LSTM efficiency
        packed = pack_padded_sequence(
            tokens, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward pass through 3 stacked bidirectional layers with dropout between layers
        packed_out1, _ = self.lstm1(packed)
        packed_out1 = torch.nn.utils.rnn.PackedSequence(
            self.dropout(packed_out1.data),
            packed_out1.batch_sizes,
            packed_out1.sorted_indices,
            packed_out1.unsorted_indices,
        )

        packed_out2, _ = self.lstm2(packed_out1)
        packed_out2 = torch.nn.utils.rnn.PackedSequence(
            self.dropout(packed_out2.data),
            packed_out2.batch_sizes,
            packed_out2.sorted_indices,
            packed_out2.unsorted_indices,
        )

        packed_out3, (hidden3, cell3) = self.lstm3(packed_out2)
        lstm_output, _ = pad_packed_sequence(packed_out3, batch_first=True)

        # Final layer states
        hidden, cell = hidden3, cell3

        # Aggregate sequence representation
        if self.config.aggregation == "last":
            # Use last hidden states from both directions (always bidirectional)
            representation = torch.cat([hidden[-2], hidden[-1]], dim=1)

        # elif self.config.aggregation == "mean":
        #     # Mean pooling over valid tokens
        #     mask = torch.arange(lstm_output.shape[1], device=tokens.device).unsqueeze(
        #         0
        #     ).expand(batch_size, -1) < lengths.unsqueeze(1)
        #     masked_output = lstm_output * mask.unsqueeze(-1)
        #     representation = masked_output.sum(dim=1) / lengths.unsqueeze(-1).float()
        #
        # elif self.config.aggregation == "max":
        #     # Max pooling over valid tokens
        #     mask = torch.arange(lstm_output.shape[1], device=tokens.device).unsqueeze(
        #         0
        #     ).expand(batch_size, -1) < lengths.unsqueeze(1)
        #     masked_output = lstm_output.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        #     representation = masked_output.max(dim=1)[0]
        #
        # elif self.config.aggregation == "attn":
        #     # Attention-based aggregation
        #     representation = self.aggregation(lstm_output, lengths)

        else:
            raise ValueError(f"Unknown aggregation method: {self.config.aggregation}")

        # Apply MLP head
        predictions = self.head(representation).squeeze(-1)  # [batch_size]

        return predictions

    def predict_and_optionally_clamp(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        clamp_for_metrics: bool = True,
    ) -> torch.Tensor:
        """Forward pass with optional clamping for metrics computation.

        During training, we don't clamp to preserve gradients.
        During evaluation, we clamp for proper metric computation.
        """
        predictions = self.forward(tokens, lengths)

        if clamp_for_metrics:
            min_val, max_val = self.config.output_range
            predictions = torch.clamp(predictions, min_val, max_val)

        return predictions


# Metrics and Loss Functions
class MetricsAccumulator:
    """Running metrics accumulator to reduce memory usage."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.predictions: list[float] = []
        self.targets: list[float] = []
        self.ids: list[str] = []

    def update(
        self, preds: torch.Tensor, targets: torch.Tensor, ids: list[str]
    ) -> None:
        """Update with batch predictions and targets."""
        self.predictions.extend(preds.detach().cpu().numpy())
        self.targets.extend(targets.detach().cpu().numpy())
        self.ids.extend(ids)

    def compute_metrics(
        self, target_scaler: TargetScaler | None = None
    ) -> dict[str, float]:
        """Compute all regression metrics (matching BERT script style)."""
        if not self.predictions:
            return {}

        preds = np.array(self.predictions)
        targets = np.array(self.targets)

        # Inverse transform if scaler was used
        if target_scaler and target_scaler.fitted and target_scaler.mode != "none":
            preds = target_scaler.inverse_transform(preds)
            targets = target_scaler.inverse_transform(targets)

        # Clamp predictions to valid range for metrics
        preds_clamped = np.clip(preds, ScoreConstants.MIN, ScoreConstants.MAX)

        # Standard regression metrics
        mae = np.mean(np.abs(preds_clamped - targets))
        mse = np.mean((preds_clamped - targets) ** 2)
        rmse = np.sqrt(mse)

        # R-squared
        ss_res = np.sum((targets - preds_clamped) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Step-aligned metrics (snap to 40-point increments)
        preds_snapped = np.array([snap_to_step(p) for p in preds_clamped])
        targets_snapped = np.array([snap_to_step(t) for t in targets])

        step_accuracy = np.mean(preds_snapped == targets_snapped)
        mae_step = np.mean(np.abs(preds_snapped - targets_snapped))

        # Round to C1 levels for kappa calculations
        true_labels_rounded = round_to_c1_levels(targets.tolist())
        predictions_rounded = round_to_c1_levels(preds_clamped.tolist())

        # Cohen's Kappa
        try:
            kappa = sklearn.metrics.cohen_kappa_score(
                true_labels_rounded, predictions_rounded
            )
        except Exception:
            kappa = 0.0

        # Quadratic Weighted Kappa
        try:
            qwk = quadratic_weighted_kappa(
                true_labels_rounded,
                predictions_rounded,
                labels=[0, 40, 80, 120, 160, 200],
            )
        except Exception:
            qwk = 0.0

        # Pearson correlation
        try:
            pearson_corr, pearson_p = scipy.stats.pearsonr(targets, preds_clamped)
            if np.isnan(pearson_corr):
                pearson_corr = 0.0
        except Exception:
            pearson_corr = 0.0

        return {
            "loss": float(mse),  # Use MAE as loss like the CLaRiCe paper
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "kappa": float(kappa),
            "qwk": float(qwk),
            "pearson_corr": float(pearson_corr),
            "step_accuracy": float(step_accuracy),
            "mae_step": float(mae_step),
            "count": len(preds),
        }

    def get_predictions_df(
        self, target_scaler: TargetScaler | None = None
    ) -> list[dict[str, str | float | int]]:
        """Get predictions as list of dictionaries for DataFrame creation."""
        if not self.predictions:
            return []

        preds = np.array(self.predictions)
        targets = np.array(self.targets)

        # Inverse transform if scaler was used
        if target_scaler and target_scaler.fitted and target_scaler.mode != "none":
            preds = target_scaler.inverse_transform(preds)
            targets = target_scaler.inverse_transform(targets)

        # Clamp and snap predictions
        preds_clamped = np.clip(preds, ScoreConstants.MIN, ScoreConstants.MAX)
        preds_snapped = np.array([snap_to_step(p) for p in preds_clamped])

        return [
            {
                "id": id_,
                "target": float(target),
                "pred": float(pred),
                "pred_snapped": int(pred_snap),
            }
            for id_, target, pred, pred_snap in zip(
                self.ids, targets, preds_clamped, preds_snapped
            )
        ]


def get_loss_fn(loss_type: Literal["mae", "mse", "huber"] = "mae") -> nn.Module:
    """Get loss function by name."""
    if loss_type == "mae":
        return nn.L1Loss()
    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type == "huber":
        return nn.HuberLoss()

    # raise ValueError(f"Unknown loss type: {loss_type}")


# Quick evaluation functions
def evaluate_model(
    model: BiLSTMRegressor,
    data_loader: DataLoader,
    device: torch.device,
    target_scaler: TargetScaler | None = None,
) -> tuple[dict[str, float], list[dict[str, str | float | int]]]:
    """Evaluate model on a dataset."""
    model.eval()
    metrics = MetricsAccumulator()

    with torch.no_grad():
        for batch in data_loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            targets = batch["targets"]
            ids = batch["ids"]

            # Get predictions (with clamping for metrics)
            if device.type == "cuda":
                with torch.autocast("cuda", enabled=True):
                    preds = model.predict_and_optionally_clamp(
                        tokens, lengths, clamp_for_metrics=True
                    )
            else:
                preds = model.predict_and_optionally_clamp(
                    tokens, lengths, clamp_for_metrics=True
                )

            metrics.update(preds, targets, ids)

    computed_metrics = metrics.compute_metrics(target_scaler)
    predictions = metrics.get_predictions_df(target_scaler)

    return computed_metrics, predictions


def create_synthetic_dataset(
    n_samples: int = 256, min_len: int = 5, max_len: int = 200
) -> EssayDataset:
    """Create a synthetic dataset for testing."""
    # Generate random embeddings and scores
    arrays = []
    scores = []
    valid_scores = [0, 40, 80, 120, 160, 200]

    for i in range(n_samples):
        seq_len = np.random.randint(min_len, max_len + 1)
        embedding = np.random.randn(seq_len, 768).astype(np.float32)
        score = float(np.random.choice(valid_scores))

        arrays.append(embedding)
        scores.append(score)

    return EssayDataset.from_memory(arrays, scores)
