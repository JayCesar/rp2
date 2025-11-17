"""1D Convolutional Neural Network for Essay C1 Score Prediction

This module implements a clean, configurable 1D CNN for predicting ENEM essay C1 scores
from BERTimbau embeddings or grammar error features.

Architecture based on Table 4 (Comp. 1) from the CLaRiCe paper:
- Convolutional layers: 28/39 filters with configurable kernel sizes
- Dense layer: 90 neurons
- Dropout: 0.303
- AdamW optimizer with learning rate 7.06e-03 and weight decay 6.61e-04

Key improvements over BiLSTM implementation:
- Dynamic layer construction with nn.ModuleList
- Cleaner forward pass without PackedSequence complexity
- Mask-aware global pooling for variable-length sequences
- No dead/commented code
- Clear separation of concerns

Data Format:
- Vectorized essays: [batch_size, seq_len, 768] BERTimbau token embeddings
- Grammar features: [batch_size, num_features] error counts from LanguageTool
- Targets: C1 scores in range [0, 200] with 40-point increments

Usage:
    from conv1d import Conv1DRegressor, ModelConfig, TrainConfig

    config = ModelConfig(conv_filters=[28, 39], dense_neurons=90)
    model = Conv1DRegressor(config)
    predictions = model(essay_embeddings, lengths)
"""

import pathlib
import re
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import polars as pl
import sklearn.model_selection
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Import shared utilities using relative imports
try:
    from ..common import (
        MetricsAccumulator,
        TargetScaler,
        ensure_dir,
        get_device,
        set_seed,
    )
except ImportError:
    # Fallback for direct script execution
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from common import (
        MetricsAccumulator,
        TargetScaler,
        ensure_dir,
        get_device,
        set_seed,
    )


# Configuration dataclasses
@dataclass
class ModelConfig:
    """Configuration for the Conv1D model architecture.

    Based on Table 4 (Comp. 1) specifications with configurable options.
    """

    # Convolutional layers
    conv_filters: list[int] = None  # Default: [28, 39]
    kernel_sizes: list[int] = None  # Default: [3, 3]
    input_dim: int = 768  # BERTimbau embedding dimension or number of features

    # Regularization
    dropout: float = 0.303  # From Table 4 (3.03e-01)

    # Dense layers
    dense_neurons: int = 90  # From Table 4

    # Pooling strategy
    pooling: Literal["max", "avg", "both"] = "max"

    # Output configuration
    output_range: tuple[int, int] = (0, 200)

    def __post_init__(self):
        """Set defaults and validate configuration."""
        if self.conv_filters is None:
            self.conv_filters = [28, 39]
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 3]

        if len(self.conv_filters) != len(self.kernel_sizes):
            raise ValueError(
                f"conv_filters ({len(self.conv_filters)}) and kernel_sizes "
                f"({len(self.kernel_sizes)}) must have same length"
            )

    @property
    def num_conv_layers(self) -> int:
        """Number of convolutional layers."""
        return len(self.conv_filters)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)


@dataclass
class TrainConfig:
    """Configuration for training parameters.

    Based on Table 4 (Comp. 1) with AdamW instead of Adam+L2.
    """

    epochs: int = 50
    batch_size: int = 32
    lr: float = 7.06e-03  # From Table 4 (7,06e-03)
    weight_decay: float = 6.61e-04  # From Table 4 L2 regularization (6,61e-04)
    optimizer: Literal["adamw"] = "adamw"
    early_stopping_patience: int = 3
    use_amp: bool = True  # Mixed precision training
    amp_dtype: torch.dtype = torch.bfloat16
    target_scaler: Literal["none", "minmax", "standard"] = "none"
    seed: int = 42
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    grad_clip_norm: float = 1.0

    def to_dict(self) -> dict:
        result = asdict(self)
        # Handle non-serializable dtype
        result["amp_dtype"] = str(self.amp_dtype)
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        # Convert amp_dtype string back to torch.dtype if needed
        if isinstance(d.get("amp_dtype"), str):
            dtype_str = d["amp_dtype"]
            if "bfloat16" in dtype_str:
                d["amp_dtype"] = torch.bfloat16
            elif "float16" in dtype_str:
                d["amp_dtype"] = torch.float16
        return cls(**d)


@dataclass
class SerializationConfig:
    """Configuration for model checkpointing and saving."""

    output_dir: pathlib.Path = pathlib.Path("runs/conv1d")
    save_best_only: bool = True
    keep_last_k: int = 3

    def to_dict(self) -> dict:
        result = asdict(self)
        result["output_dir"] = str(self.output_dir)
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "SerializationConfig":
        if isinstance(d.get("output_dir"), str):
            d["output_dir"] = pathlib.Path(d["output_dir"])
        return cls(**d)


# Mask-aware pooling utilities (optimized)
def masked_maxpool_1d(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Apply max pooling over time dimension with masking for padding.

    Optimized to minimize allocations and use efficient broadcasting.

    Args:
        x: Input tensor [batch_size, channels, seq_len]
        lengths: Valid sequence lengths [batch_size]

    Returns:
        Pooled tensor [batch_size, channels]
    """
    # Create mask efficiently using view and comparison
    # Shape: [1, 1, seq_len] vs [batch_size, 1, 1]
    mask = torch.arange(x.shape[2], device=x.device, dtype=torch.long).view(
        1, 1, -1
    ) < lengths.view(-1, 1, 1)

    # Set padded positions to -inf and return max (ignore indices)
    return x.masked_fill(~mask, float("-inf")).max(dim=2)[0]


def masked_avgpool_1d(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Apply average pooling over time dimension with masking for padding.

    Optimized to use efficient operations and minimize memory.

    Args:
        x: Input tensor [batch_size, channels, seq_len]
        lengths: Valid sequence lengths [batch_size]

    Returns:
        Pooled tensor [batch_size, channels]
    """
    # Create mask efficiently
    mask = torch.arange(x.shape[2], device=x.device, dtype=torch.long).view(
        1, 1, -1
    ) < lengths.view(-1, 1, 1)

    # Sum and divide by lengths (clamped to avoid division by zero)
    return (x * mask).sum(dim=2) / lengths.view(-1, 1).clamp(min=1).float()


# Model architecture
class Conv1DRegressor(nn.Module):
    """1D Convolutional Neural Network for essay score regression.

    Architecture:
    - Stack of Conv1D layers with BatchNorm, ReLU, and Dropout
    - Global max/avg pooling to aggregate sequence
    - MLP head for regression

    Handles two input modes:
    1. Token sequences [B, L, D]: For BERTimbau embeddings
    2. Feature vectors [B, F]: For grammar error counts
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Build convolutional layers dynamically
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(config.num_conv_layers):
            in_channels = config.input_dim if i == 0 else config.conv_filters[i - 1]
            out_channels = config.conv_filters[i]
            kernel_size = config.kernel_sizes[i]

            # Compute padding for "same" convolution
            padding = (kernel_size - 1) // 2

            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,  # BatchNorm handles bias
            )

            self.conv_layers.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(out_channels))
            self.dropouts.append(nn.Dropout(config.dropout))

        # Compute pooled feature dimension
        last_conv_channels = config.conv_filters[-1]
        if config.pooling == "both":
            pooled_dim = last_conv_channels * 2
        else:
            pooled_dim = last_conv_channels

        # Dense head for regression
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, config.dense_neurons),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dense_neurons, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor
                - For sequences: [batch_size, seq_len, input_dim]
                - For features: [batch_size, input_dim]
            lengths: Valid sequence lengths [batch_size] (None for features)

        Returns:
            Predictions [batch_size]
        """
        # Handle different input formats
        if x.dim() == 2:
            # Feature vector: [B, F] -> [B, F, 1]
            x = x.unsqueeze(2)
            is_sequence = False
        elif x.dim() == 3:
            # Token sequence: [B, L, D] -> [B, D, L]
            x = x.transpose(1, 2)
            is_sequence = True
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

        # Pass through convolutional layers
        for conv, bn, dropout in zip(self.conv_layers, self.batch_norms, self.dropouts):
            x = conv(x)
            x = bn(x)
            x = torch.relu(x)
            x = dropout(x)

        # Global pooling
        if is_sequence and lengths is not None:
            # Mask-aware pooling for variable-length sequences
            if self.config.pooling == "max":
                x = masked_maxpool_1d(x, lengths)
            elif self.config.pooling == "avg":
                x = masked_avgpool_1d(x, lengths)
            elif self.config.pooling == "both":
                x_max = masked_maxpool_1d(x, lengths)
                x_avg = masked_avgpool_1d(x, lengths)
                x = torch.cat([x_max, x_avg], dim=1)
        else:
            # Standard pooling for features or when no lengths provided
            if self.config.pooling == "max":
                x, _ = torch.max(x, dim=2)
            elif self.config.pooling == "avg":
                x = torch.mean(x, dim=2)
            elif self.config.pooling == "both":
                x_max, _ = torch.max(x, dim=2)
                x_avg = torch.mean(x, dim=2)
                x = torch.cat([x_max, x_avg], dim=1)

        # Regression head
        predictions = self.head(x).squeeze(-1)  # [batch_size]

        return predictions


# Dataset implementation
class EssayDataset(Dataset):
    """Dataset for essay C1 score prediction with auto input-type detection.

    Supports two modes:
    1. Vectorized essays: BERTimbau token embeddings [seq_len, 768]
    2. Grammar features: Error counts from LanguageTool [num_features]
    """

    def __init__(self, data: pl.DataFrame):
        super().__init__()
        self.data = data

        # Detect input mode
        cols = set(self.data.columns)
        self.is_sequence = "essay_token_embeddings" in cols

        if self.is_sequence:
            self.feature_cols = []
        else:
            # Extract SCREAMING_SNAKE_CASE feature columns
            pattern = re.compile(r"^[A-Z0-9_]+$")
            self.feature_cols = [
                c
                for c in self.data.columns
                if c.lower() != "c1" and pattern.match(c) and c.lower() != "id"
            ]

            if not self.feature_cols:
                raise ValueError("No feature columns found for feature mode")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.row(idx, named=True)

        if self.is_sequence:
            # Mode 1: Token embeddings
            token_embeddings = np.array(row["essay_token_embeddings"], dtype=np.float32)
            tokens = torch.from_numpy(token_embeddings)  # [seq_len, 768]

            # Calculate actual sequence length (non-zero rows)
            lengths = len(token_embeddings)

            return {
                "id": f"essay_{idx}",
                "tokens": tokens,
                "lengths": torch.tensor(lengths, dtype=torch.long),
                "targets": torch.tensor(row["c1"], dtype=torch.float32),
            }
        else:
            # Mode 2: Feature vector
            features = [float(row.get(c, 0.0)) for c in self.feature_cols]
            tokens = torch.tensor(features, dtype=torch.float32)

            return {
                "id": f"essay_{idx}",
                "tokens": tokens,
                "lengths": None,
                "targets": torch.tensor(row["c1"], dtype=torch.float32),
            }


def collate_batch(batch: list[dict]) -> dict:
    """Collate function for batching essay data.

    Handles both sequence and feature inputs with proper padding.
    """
    ids = [item["id"] for item in batch]
    targets = torch.stack([item["targets"] for item in batch])

    # Check if we have sequences or features
    if batch[0]["lengths"] is not None:
        # Sequence mode: pad to max length
        tokens = [item["tokens"] for item in batch]
        lengths = torch.stack([item["lengths"] for item in batch])

        # Pad sequences
        from torch.nn.utils.rnn import pad_sequence

        padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=0.0)

        return {
            "ids": ids,
            "tokens": padded_tokens,
            "lengths": lengths,
            "targets": targets,
        }
    else:
        # Feature mode: just stack
        tokens = torch.stack([item["tokens"] for item in batch])

        return {
            "ids": ids,
            "tokens": tokens,
            "lengths": None,
            "targets": targets,
        }


def create_data_loader(
    dataset: EssayDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Create DataLoader with appropriate settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def split_dataset(
    df: pl.DataFrame,
    val_ratio: float = 0.10,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split dataset into train/val/test with stratification by C1 scores.

    Uses 20-point bins for stratification to ensure balanced score distribution.

    Args:
        df: DataFrame with 'c1' column
        val_ratio: Validation set proportion
        test_ratio: Test set proportion
        seed: Random seed for reproducibility

    Returns:
        (train_df, val_df, test_df)
    """
    # Extract C1 scores and create stratification bins
    scores = df["c1"].to_numpy()
    indices = np.arange(len(df))

    # Create bins: 0-19, 20-39, ..., 180-200
    bins = np.clip(scores // 20, 0, 10).astype(int)

    # First split: train vs (val + test)
    train_indices, temp_indices = sklearn.model_selection.train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=bins,
    )

    # Second split: val vs test
    temp_scores = scores[temp_indices]
    temp_bins = np.clip(temp_scores // 20, 0, 10).astype(int)

    val_proportion = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = sklearn.model_selection.train_test_split(
        temp_indices,
        test_size=(1 - val_proportion),
        random_state=seed,
        stratify=temp_bins,
    )

    # Create split DataFrames
    train_df = df[train_indices]
    val_df = df[val_indices]
    test_df = df[test_indices]

    return train_df, val_df, test_df
