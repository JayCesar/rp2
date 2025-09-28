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

import argparse
import json
import logging
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal
from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset, random_split

# Constants
class ScoreConstants:
    MIN = 0
    MAX = 200
    STEP = 40

# Configure logging
def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> None:
    """Set up logging with console and optional file output."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=handlers
    )
    
    # Reduce noise from some libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Configuration dataclasses
@dataclass
class ModelConfig:
    """Configuration for the BiLSTM model architecture."""
    input_dim: int = 768
    hidden_size: int = 256
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.1
    aggregation: Literal['last', 'mean', 'max', 'attn'] = 'last'
    mlp_hidden: int | None = 256
    use_layer_norm: bool = False
    output_range: tuple[int, int] = (ScoreConstants.MIN, ScoreConstants.MAX)
    
    def to_dict(self) -> dict[str, int | float | bool | str | tuple[int, int]]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict[str, int | float | bool | str | tuple[int, int]]) -> 'ModelConfig':
        return cls(**d)

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    train_csv: str | None = None
    val_csv: str | None = None
    test_csv: str | None = None
    id_column: str = 'id'
    embedding_column: str = 'embedding_path'
    score_column: str = 'c1'
    embedding_format: Literal['npy', 'pt', 'auto'] = 'auto'
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
    def from_dict(cls, d: dict[str, str | int | float | bool | None]) -> 'DataConfig':
        return cls(**d)

@dataclass
class TrainConfig:
    """Configuration for training parameters."""
    epochs: int = 20
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    optimizer: Literal['adamw'] = 'adamw'
    scheduler: Literal['plateau', 'onecycle', 'none'] = 'plateau'
    plateau_patience: int = 3
    plateau_factor: float = 0.5
    onecycle_pct_start: float = 0.1
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    seed: int = 42
    device: Literal['auto', 'cpu', 'cuda', 'mps'] = 'auto'
    use_amp: bool = True
    amp_dtype: Literal['bf16', 'fp16'] = 'bf16'
    compile: bool = False
    target_scaler: Literal['none', 'minmax', 'standard'] = 'minmax'
    
    def to_dict(self) -> dict[str, str | int | float | bool]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict[str, str | int | float | bool]) -> 'TrainConfig':
        return cls(**d)

@dataclass
class SerializationConfig:
    """Configuration for model checkpointing and saving."""
    output_dir: str = 'runs/bilstm'
    save_best_only: bool = True
    keep_last_k: int = 3
    
    def to_dict(self) -> dict[str, str | int | bool]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict[str, str | int | bool]) -> 'SerializationConfig':
        return cls(**d)

# Custom Exceptions
class DataFormatError(Exception):
    """Raised when data format is invalid or incompatible."""
    pass

class ModelConfigError(Exception):
    """Raised when model configuration is invalid."""
    pass

# Utility functions
def get_device(preference: str = 'auto') -> torch.device:
    """Auto-detect or select the best available device."""
    if preference != 'auto':
        device = torch.device(preference)
        logger.info(f"Using specified device: {device}")
        return device
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # Check AMP support
    if device.type == 'cuda':
        if torch.cuda.get_device_capability(device)[0] >= 8:  # Ampere+
            logger.info("BFloat16 AMP available (recommended)")
        else:
            logger.info("Float16 AMP available")
    
    return device

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # For RNN performance, we don't want full determinism
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    logger.info(f"Random seed set to {seed}")

def ensure_dir(path: str | Path) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def safe_float(value: str | int | float, field_name: str) -> float:
    """Safely convert value to float with informative error message."""
    try:
        return float(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not convert {field_name}='{value}' to float: {e}")

def snap_to_step(x: float, step: int = ScoreConstants.STEP, 
                 min_val: int = ScoreConstants.MIN, 
                 max_val: int = ScoreConstants.MAX) -> int:
    """Round to nearest step and clamp to valid range."""
    snapped = round(x / step) * step
    return max(min_val, min(max_val, int(snapped)))

# Target scaling utilities
class TargetScaler:
    """Scale targets for improved training stability."""
    
    def __init__(self, mode: str = 'minmax') -> None:
        self.mode: str = mode
        self.fitted: bool = False
        self.mean_: float | None = None
        self.std_: float | None = None
        self.min_: float | None = None
        self.max_: float | None = None
    
    def fit(self, y: np.ndarray) -> 'TargetScaler':
        """Fit scaler to target values."""
        y = np.array(y)
        
        if self.mode == 'minmax':
            self.min_ = float(ScoreConstants.MIN)
            self.max_ = float(ScoreConstants.MAX)
        elif self.mode == 'standard':
            self.mean_ = float(np.mean(y))
            self.std_ = float(np.std(y))
            if self.std_ == 0:
                self.std_ = 1.0  # Avoid division by zero
        elif self.mode == 'none':
            pass
        else:
            raise ValueError(f"Unknown scaling mode: {self.mode}")
        
        self.fitted = True
        return self
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform targets using fitted scaler."""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        y = np.array(y, dtype=np.float32)
        
        if self.mode == 'minmax':
            return (y - self.min_) / (self.max_ - self.min_)
        elif self.mode == 'standard':
            return (y - self.mean_) / self.std_
        elif self.mode == 'none':
            return y
        else:
            raise ValueError(f"Unknown scaling mode: {self.mode}")
    
    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform scaled targets back to original scale."""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        
        y_scaled = np.array(y_scaled, dtype=np.float32)
        
        if self.mode == 'minmax':
            return y_scaled * (self.max_ - self.min_) + self.min_
        elif self.mode == 'standard':
            return y_scaled * self.std_ + self.mean_
        elif self.mode == 'none':
            return y_scaled
        else:
            raise ValueError(f"Unknown scaling mode: {self.mode}")
    
    def state_dict(self) -> dict[str, str | bool | float | None]:
        """Return state dictionary for serialization."""
        return {
            'mode': self.mode,
            'fitted': self.fitted,
            'mean_': self.mean_,
            'std_': self.std_,
            'min_': self.min_,
            'max_': self.max_
        }
    
    def load_state_dict(self, state_dict: dict[str, str | bool | float | None]) -> None:
        """Load state from dictionary."""
        self.mode = state_dict['mode']
        self.fitted = state_dict['fitted']
        self.mean_ = state_dict['mean_']
        self.std_ = state_dict['std_']
        self.min_ = state_dict['min_']
        self.max_ = state_dict['max_']

# Data loading utilities
@dataclass
class DataRecord:
    """A single data record with embeddings and target score."""
    id: str
    path: str | None = None
    array: np.ndarray | None = None
    score: float = 0.0
    
    def __post_init__(self) -> None:
        if (self.path is None) == (self.array is None):
            raise ValueError("Exactly one of 'path' or 'array' must be provided")

def load_embedding(path: str, fmt: str = 'auto') -> torch.FloatTensor:
    """Load embedding file and return as torch tensor."""
    try:
        if fmt == 'auto':
            if path.endswith('.npy'):
                fmt = 'npy'
            elif path.endswith('.pt') or path.endswith('.pth'):
                fmt = 'pt'
            else:
                raise DataFormatError(f"Cannot auto-detect format for {path}. Use explicit format.")
        
        if fmt == 'npy':
            array = np.load(path)
            tensor = torch.from_numpy(array).float()
        elif fmt == 'pt':
            tensor = torch.load(path, map_location='cpu')
            if not isinstance(tensor, torch.Tensor):
                raise DataFormatError(f"Expected torch.Tensor in {path}, got {type(tensor)}")
            tensor = tensor.float()
        else:
            raise DataFormatError(f"Unsupported format: {fmt}")
        
        # Validate shape
        if tensor.ndim != 2:
            raise DataFormatError(f"Expected 2D tensor [seq_len, 768] in {path}, got shape {tensor.shape}")
        if tensor.shape[-1] != 768:
            raise DataFormatError(f"Expected 768 features in {path}, got {tensor.shape[-1]}")
        
        return tensor
    
    except Exception as e:
        raise DataFormatError(f"Failed to load embedding from {path}: {e}")

def maybe_truncate(tensor: torch.Tensor, max_len: int, strategy: str = 'head') -> torch.Tensor:
    """Truncate sequence if it exceeds max_len."""
    if len(tensor) <= max_len:
        return tensor
    
    if strategy == 'head':
        return tensor[:max_len]
    elif strategy == 'center':
        start = (len(tensor) - max_len) // 2
        return tensor[start:start + max_len]
    elif strategy == 'head_tail':
        head_len = max_len // 2
        tail_len = max_len - head_len
        return torch.cat([tensor[:head_len], tensor[-tail_len:]], dim=0)
    else:
        raise ValueError(f"Unknown truncation strategy: {strategy}")

class EmbeddingSequenceDataset(Dataset):
    """Dataset for loading embedding sequences and target scores."""
    
    def __init__(self, records: list[DataRecord], embedding_format: str = 'auto',
                 max_seq_len: int = 1024, pad_value: float = 0.0) -> None:
        self.records: list[DataRecord] = records
        self.embedding_format: str = embedding_format
        self.max_seq_len: int = max_seq_len
        self.pad_value: float = pad_value
        
        # Log sequence length statistics
        if records:
            self._log_sequence_stats()
    
    def _log_sequence_stats(self) -> None:
        """Log sequence length statistics for the dataset."""
        lengths: list[int] = []
        failed_count: int = 0
        
        # Sample a subset to avoid loading all embeddings
        sample_size = min(100, len(self.records))
        sample_indices = np.random.choice(len(self.records), sample_size, replace=False)
        
        for idx in sample_indices:
            try:
                tokens = self._load_tokens(idx)
                lengths.append(len(tokens))
            except Exception:
                failed_count += 1
        
        if lengths:
            lengths = np.array(lengths)
            logger.info(f"Sequence length stats (n={len(lengths)}): "
                       f"mean={np.mean(lengths):.1f}, "
                       f"std={np.std(lengths):.1f}, "
                       f"min={np.min(lengths)}, "
                       f"max={np.max(lengths)}, "
                       f"median={np.median(lengths):.1f}")
            
            if np.max(lengths) > self.max_seq_len:
                truncated_pct = np.sum(lengths > self.max_seq_len) / len(lengths) * 100
                logger.info(f"{truncated_pct:.1f}% of sequences will be truncated (max_seq_len={self.max_seq_len})")
        
        if failed_count > 0:
            logger.warning(f"Failed to load {failed_count} sample embeddings")
    
    def _load_tokens(self, idx: int) -> torch.Tensor:
        """Load tokens for a given index."""
        record = self.records[idx]
        
        if record.array is not None:
            tokens = torch.from_numpy(record.array).float()
        else:
            tokens = load_embedding(record.path, self.embedding_format)
        
        # Truncate if necessary
        if len(tokens) > self.max_seq_len:
            tokens = maybe_truncate(tokens, self.max_seq_len)
        
        # Handle empty sequences
        if len(tokens) == 0:
            logger.warning(f"Empty sequence for record {record.id}, using single zero vector")
            tokens = torch.zeros(1, 768, dtype=torch.float32)
        
        return tokens
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | float | str]:
        """Get a single data item."""
        record: DataRecord = self.records[idx]
        
        try:
            tokens = self._load_tokens(idx)
        except Exception as e:
            logger.error(f"Failed to load tokens for {record.id}: {e}")
            # Return a single zero vector as fallback
            tokens = torch.zeros(1, 768, dtype=torch.float32)
        
        return {
            'tokens': tokens,
            'target': record.score,
            'id': record.id
        }
    
    @classmethod
    def from_csv(cls, csv_path: str, id_column: str, embedding_column: str,
                 score_column: str, embedding_format: str = 'auto',
                 max_seq_len: int = 1024, pad_value: float = 0.0,
                 filters: dict[str, str | list[str]] | None = None) -> 'EmbeddingSequenceDataset':
        """Create dataset from CSV file."""
        import pandas as pd
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise DataFormatError(f"Failed to read CSV {csv_path}: {e}")
        
        # Validate required columns
        required_cols = [id_column, embedding_column, score_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataFormatError(f"Missing columns in {csv_path}: {missing_cols}")
        
        # Apply filters if provided
        if filters:
            for col, values in filters.items():
                if col in df.columns:
                    df = df[df[col].isin(values if isinstance(values, list) else [values])]
        
        records: list[DataRecord] = []
        skipped: int = 0
        
        for _, row in df.iterrows():
            try:
                record = DataRecord(
                    id=str(row[id_column]),
                    path=str(row[embedding_column]),
                    score=safe_float(row[score_column], score_column)
                )
                records.append(record)
            except Exception as e:
                logger.warning(f"Skipping row with ID {row.get(id_column, 'unknown')}: {e}")
                skipped += 1
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} invalid records from {csv_path}")
        
        logger.info(f"Loaded {len(records)} records from {csv_path}")
        
        return cls(records, embedding_format, max_seq_len, pad_value)
    
    @classmethod
    def from_memory(cls, arrays: list[np.ndarray], scores: list[float],
                    ids: list[str] | None = None,
                    embedding_format: str = 'auto', max_seq_len: int = 1024,
                    pad_value: float = 0.0) -> 'EmbeddingSequenceDataset':
        """Create dataset from in-memory arrays."""
        if len(arrays) != len(scores):
            raise ValueError("Arrays and scores must have same length")
        
        if ids is None:
            ids = [f"sample_{i}" for i in range(len(arrays))]
        elif len(ids) != len(arrays):
            raise ValueError("IDs must have same length as arrays")
        
        records = [
            DataRecord(id=id_, array=array, score=score)
            for id_, array, score in zip(ids, arrays, scores)
        ]
        
        return cls(records, embedding_format, max_seq_len, pad_value)

def collate_batch(batch: list[dict[str, torch.Tensor | float | str]], pad_value: float = 0.0) -> dict[str, torch.Tensor | list[str]]:
    """Collate function for DataLoader."""
    tokens_list: list[torch.Tensor] = []
    lengths: list[int] = []
    targets: list[float] = []
    ids: list[str] = []
    
    for item in batch:
        tokens = item['tokens']
        
        # Ensure minimum length of 1
        if len(tokens) == 0:
            tokens = torch.zeros(1, 768, dtype=torch.float32)
        
        tokens_list.append(tokens)
        lengths.append(len(tokens))
        targets.append(item['target'])
        ids.append(item['id'])
    
    # Pad sequences
    tokens_padded = pad_sequence(tokens_list, batch_first=True, padding_value=pad_value)
    lengths = torch.tensor(lengths, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.float32)
    
    return {
        'tokens': tokens_padded,
        'lengths': lengths,
        'targets': targets,
        'ids': ids
    }

def split_dataset(dataset: Dataset, val_ratio: float, test_ratio: float = 0.0, 
                  seed: int = 42) -> tuple[Subset, Subset | None, Subset | None]:
    """Split dataset into train/val/test subsets."""
    dataset_size = len(dataset)
    
    # Calculate split sizes
    test_size = int(dataset_size * test_ratio)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size - test_size
    
    if train_size <= 0:
        raise ValueError("Train set would be empty. Reduce val_ratio or test_ratio.")
    
    # Create splits
    generator = torch.Generator().manual_seed(seed)
    
    if test_size > 0:
        train_val_dataset, test_dataset = random_split(
            dataset, [train_size + val_size, test_size], generator=generator
        )
        train_dataset, val_dataset = random_split(
            train_val_dataset, [train_size, val_size], generator=generator
        )
        return train_dataset, val_dataset, test_dataset
    else:
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )
        return train_dataset, val_dataset, None

# Model implementation
class BiLSTMRegressor(nn.Module):
    """Bidirectional LSTM for regression with flexible aggregation strategies."""
    
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config: ModelConfig = config
        
        # Validate input dimension
        if config.input_dim != 768:
            logger.warning(f"Input dimension {config.input_dim} != 768. "
                          f"Make sure this matches your embeddings.")
        
        # LSTM layer
        lstm_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=config.bidirectional
        )
        
        # Calculate representation dimension
        self.rep_dim = config.hidden_size * (2 if config.bidirectional else 1)
        
        # Attention mechanism for 'attn' aggregation
        if config.aggregation == 'attn':
            self.attention_query = nn.Parameter(torch.randn(self.rep_dim))
            self.attention_proj = nn.Linear(self.rep_dim, 1)
        
        # Optional layer normalization
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.rep_dim)
        else:
            self.layer_norm = None
        
        # MLP head
        if config.mlp_hidden is not None:
            self.head = nn.Sequential(
                nn.Linear(self.rep_dim, config.mlp_hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.mlp_hidden, 1)
            )
        else:
            self.head = nn.Linear(self.rep_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0.0)
                    # Set forget gate bias to 1
                    n = param.size(0)
                    param.data[(n//4):(n//2)].fill_(1.0)
            elif 'head' in name and 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'head' in name and 'bias' in name:
                param.data.fill_(0.0)
    
    def _aggregate_last(self, lstm_output: torch.Tensor | None, hidden: tuple[torch.Tensor, torch.Tensor],
                       lengths: torch.Tensor) -> torch.Tensor:
        """Use the last hidden states from each direction."""
        h_n, _ = hidden  # [num_layers * num_directions, batch, hidden_size]
        
        if self.config.bidirectional:
            # Concatenate forward and backward final hidden states
            # h_n[-2] is the forward direction of the last layer
            # h_n[-1] is the backward direction of the last layer
            representation = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            # Use the final hidden state from the last layer
            representation = h_n[-1]
        
        return representation  # [batch, rep_dim]
    
    def _aggregate_pooling(self, lstm_output: torch.Tensor, lengths: torch.Tensor,
                          method: str = 'mean') -> torch.Tensor:
        """Apply mean or max pooling over the sequence dimension."""
        batch_size, max_len, hidden_size = lstm_output.shape
        
        # Create mask for valid positions
        mask = torch.arange(max_len, device=lstm_output.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(-1).expand_as(lstm_output)  # [batch, max_len, hidden_size]
        
        if method == 'mean':
            # Masked mean pooling
            masked_output = lstm_output * mask.float()
            representation = masked_output.sum(dim=1) / lengths.float().unsqueeze(-1)
        elif method == 'max':
            # Masked max pooling
            masked_output = lstm_output.masked_fill(~mask, float('-inf'))
            representation, _ = masked_output.max(dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {method}")
        
        return representation  # [batch, rep_dim]
    
    def _aggregate_attention(self, lstm_output: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Apply attention mechanism over the sequence."""
        batch_size, max_len, hidden_size = lstm_output.shape
        
        # Compute attention scores
        # lstm_output: [batch, max_len, hidden_size]
        # attention_query: [hidden_size]
        attention_scores = torch.matmul(lstm_output, self.attention_query)  # [batch, max_len]
        
        # Create mask for valid positions
        mask = torch.arange(max_len, device=lstm_output.device)[None, :] < lengths[:, None]
        
        # Apply mask to attention scores
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch, max_len]
        
        # Weighted sum of LSTM outputs
        representation = torch.sum(
            lstm_output * attention_weights.unsqueeze(-1), dim=1
        )  # [batch, hidden_size]
        
        return representation  # [batch, rep_dim]
    
    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            tokens: [batch_size, max_len, 768]
            lengths: [batch_size] - actual sequence lengths
            
        Returns:
            predictions: [batch_size] - regression predictions
        """
        batch_size = tokens.shape[0]
        
        # Pack sequences for efficient LSTM processing
        packed_input = pack_padded_sequence(
            tokens, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM forward pass
        packed_output, hidden = self.lstm(packed_input)
        
        # Unpack for aggregation (except for 'last' which uses hidden directly)
        if self.config.aggregation != 'last':
            lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_output = None
        
        # Apply aggregation strategy
        if self.config.aggregation == 'last':
            representation = self._aggregate_last(lstm_output, hidden, lengths)
        elif self.config.aggregation == 'mean':
            representation = self._aggregate_pooling(lstm_output, lengths, 'mean')
        elif self.config.aggregation == 'max':
            representation = self._aggregate_pooling(lstm_output, lengths, 'max')
        elif self.config.aggregation == 'attn':
            representation = self._aggregate_attention(lstm_output, lengths)
        else:
            raise ValueError(f"Unknown aggregation method: {self.config.aggregation}")
        
        # Apply layer normalization if configured
        if self.layer_norm is not None:
            representation = self.layer_norm(representation)
        
        # Apply MLP head
        predictions = self.head(representation).squeeze(-1)  # [batch_size]
        
        return predictions
    
    def predict_and_optionally_clamp(self, tokens: torch.Tensor, lengths: torch.Tensor,
                                   clamp_for_metrics: bool = True) -> torch.Tensor:
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
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, ids: list[str]) -> None:
        """Update with batch predictions and targets."""
        self.predictions.extend(preds.detach().cpu().numpy())
        self.targets.extend(targets.detach().cpu().numpy())
        self.ids.extend(ids)
    
    def compute_metrics(self, target_scaler: TargetScaler | None = None) -> dict[str, float]:
        """Compute all regression metrics."""
        if not self.predictions:
            return {}
        
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Inverse transform if scaler was used
        if target_scaler and target_scaler.fitted and target_scaler.mode != 'none':
            preds = target_scaler.inverse_transform(preds)
            targets = target_scaler.inverse_transform(targets)
        
        # Clamp predictions to valid range for metrics
        preds_clamped = np.clip(preds, ScoreConstants.MIN, ScoreConstants.MAX)
        
        # Standard regression metrics
        mae = np.mean(np.abs(preds_clamped - targets))
        rmse = np.sqrt(np.mean((preds_clamped - targets) ** 2))
        
        # R-squared
        ss_res = np.sum((targets - preds_clamped) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Step-aligned metrics (snap to 40-point increments)
        preds_snapped = np.array([snap_to_step(p) for p in preds_clamped])
        targets_snapped = np.array([snap_to_step(t) for t in targets])
        
        step_accuracy = np.mean(preds_snapped == targets_snapped)
        mae_step = np.mean(np.abs(preds_snapped - targets_snapped))
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'step_accuracy': float(step_accuracy),
            'mae_step': float(mae_step),
            'count': len(preds)
        }
    
    def get_predictions_df(self, target_scaler: TargetScaler | None = None) -> list[dict[str, str | float | int]]:
        """Get predictions as list of dictionaries for DataFrame creation."""
        if not self.predictions:
            return []
        
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Inverse transform if scaler was used
        if target_scaler and target_scaler.fitted and target_scaler.mode != 'none':
            preds = target_scaler.inverse_transform(preds)
            targets = target_scaler.inverse_transform(targets)
        
        # Clamp and snap predictions
        preds_clamped = np.clip(preds, ScoreConstants.MIN, ScoreConstants.MAX)
        preds_snapped = np.array([snap_to_step(p) for p in preds_clamped])
        
        return [
            {
                'id': id_,
                'target': float(target),
                'pred': float(pred),
                'pred_snapped': int(pred_snap)
            }
            for id_, target, pred, pred_snap in zip(
                self.ids, targets, preds_clamped, preds_snapped
            )
        ]

def get_loss_fn(loss_type: str = 'mse') -> nn.Module:
    """Get loss function by name."""
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'huber':
        return nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

# Quick evaluation functions
def evaluate_model(model: BiLSTMRegressor, data_loader: DataLoader, device: torch.device,
                   target_scaler: TargetScaler | None = None) -> tuple[dict[str, float], list[dict[str, str | float | int]]]:
    """Evaluate model on a dataset."""
    model.eval()
    metrics = MetricsAccumulator()
    
    with torch.no_grad():
        for batch in data_loader:
            tokens = batch['tokens'].to(device, non_blocking=True)
            lengths = batch['lengths'].to(device, non_blocking=True)
            targets = batch['targets']
            ids = batch['ids']
            
            # Get predictions (with clamping for metrics)
            if device.type == 'cuda':
                with autocast(enabled=True):
                    preds = model.predict_and_optionally_clamp(tokens, lengths, clamp_for_metrics=True)
            else:
                preds = model.predict_and_optionally_clamp(tokens, lengths, clamp_for_metrics=True)
            
            metrics.update(preds, targets, ids)
    
    computed_metrics = metrics.compute_metrics(target_scaler)
    predictions = metrics.get_predictions_df(target_scaler)
    
    return computed_metrics, predictions

def create_synthetic_dataset(n_samples: int = 256, min_len: int = 5, max_len: int = 200) -> EmbeddingSequenceDataset:
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
    
    return EmbeddingSequenceDataset.from_memory(arrays, scores)

def smoke_test() -> None:
    """Run a quick smoke test with synthetic data."""
    logger.info("Running smoke test with synthetic data...")
    
    # Set up
    device = get_device('auto')
    set_seed(42)
    
    # Create synthetic dataset
    dataset = create_synthetic_dataset(256)
    train_dataset, val_dataset, _ = split_dataset(dataset, 0.2, 0.0, seed=42)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, 0.0),
        num_workers=0  # Avoid multiprocessing issues in tests
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, 0.0),
        num_workers=0
    )
    
    # Create model
    model_config = ModelConfig(hidden_size=64, num_layers=1)
    model = BiLSTMRegressor(model_config).to(device)
    
    # Create optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = get_loss_fn('mse')
    
    # Train for 2 epochs
    logger.info("Training for 2 epochs...")
    for epoch in range(2):
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            tokens = batch['tokens'].to(device)
            lengths = batch['lengths'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            preds = model(tokens, lengths)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx == 0:  # Log first batch
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}: loss={loss.item():.4f}")
        
        # Validation
        val_metrics, _ = evaluate_model(model, val_loader, device)
        logger.info(f"Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}, "
                   f"val_rmse={val_metrics.get('rmse', 0):.4f}, "
                   f"val_step_acc={val_metrics.get('step_accuracy', 0):.3f}")
    
    logger.info("Smoke test completed successfully!")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bidirectional LSTM for Essay C1 Score Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with validation split
  python bidirectional_lstm.py train --train-csv data.csv --val-split 0.2
  
  # Train with separate validation file
  python bidirectional_lstm.py train --train-csv train.csv --val-csv val.csv
  
  # Evaluate model
  python bidirectional_lstm.py eval --checkpoint runs/bilstm/best.pt --test-csv test.csv
  
  # Generate predictions
  python bidirectional_lstm.py predict --checkpoint runs/bilstm/best.pt --input-csv new_data.csv
  
  # Run smoke test
  python bidirectional_lstm.py smoke
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto',
                      help='Device to use (default: auto)')
        p.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
        p.add_argument('--output-dir', type=str, default='runs/bilstm',
                      help='Output directory (default: runs/bilstm)')
    
    def add_data_args(p: argparse.ArgumentParser) -> None:
        p.add_argument('--id-col', type=str, default='id', help='ID column name')
        p.add_argument('--embedding-col', type=str, default='embedding_path',
                      help='Embedding file path column name')
        p.add_argument('--score-col', type=str, default='c1', help='Score column name')
        p.add_argument('--embedding-format', choices=['auto', 'npy', 'pt'], default='auto',
                      help='Embedding file format')
        p.add_argument('--max-seq-len', type=int, default=1024,
                      help='Maximum sequence length')
        p.add_argument('--num-workers', type=int, default=4,
                      help='Number of data loader workers')
    
    def add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument('--hidden-size', type=int, default=256,
                      help='LSTM hidden size (default: 256)')
        p.add_argument('--num-layers', type=int, default=2,
                      help='Number of LSTM layers (default: 2)')
        p.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate (default: 0.1)')
        p.add_argument('--aggregation', choices=['last', 'mean', 'max', 'attn'], default='last',
                      help='Sequence aggregation method (default: last)')
        p.add_argument('--mlp-hidden', type=int, default=256,
                      help='MLP head hidden size (default: 256)')
        p.add_argument('--layer-norm', action='store_true',
                      help='Use layer normalization')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    add_common_args(train_parser)
    add_data_args(train_parser)
    add_model_args(train_parser)
    
    train_parser.add_argument('--train-csv', type=str, required=True,
                            help='Training data CSV file')
    train_parser.add_argument('--val-csv', type=str,
                            help='Validation data CSV file (optional)')
    train_parser.add_argument('--val-split', type=float, default=0.1,
                            help='Validation split ratio if no val-csv provided')
    train_parser.add_argument('--epochs', type=int, default=20,
                            help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32,
                            help='Training batch size')
    train_parser.add_argument('--lr', type=float, default=2e-4,
                            help='Learning rate')
    train_parser.add_argument('--weight-decay', type=float, default=1e-4,
                            help='Weight decay')
    train_parser.add_argument('--scheduler', choices=['plateau', 'onecycle', 'none'],
                            default='plateau', help='Learning rate scheduler')
    train_parser.add_argument('--patience', type=int, default=5,
                            help='Early stopping patience')
    train_parser.add_argument('--grad-clip', type=float, default=1.0,
                            help='Gradient clipping norm')
    train_parser.add_argument('--target-scaler', choices=['none', 'minmax', 'standard'],
                            default='minmax', help='Target scaling method')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate the model')
    add_common_args(eval_parser)
    add_data_args(eval_parser)
    
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                           help='Model checkpoint to evaluate')
    eval_parser.add_argument('--test-csv', type=str, required=True,
                           help='Test data CSV file')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    add_common_args(predict_parser)
    add_data_args(predict_parser)
    
    predict_parser.add_argument('--checkpoint', type=str, required=True,
                              help='Model checkpoint for prediction')
    predict_parser.add_argument('--input-csv', type=str, required=True,
                              help='Input data CSV file')
    predict_parser.add_argument('--output', type=str, default='predictions.csv',
                              help='Output predictions CSV file')
    predict_parser.add_argument('--snap-to-step', action='store_true',
                              help='Include step-snapped predictions')
    
    # Smoke test command
    smoke_parser = subparsers.add_parser('smoke', help='Run smoke test with synthetic data')
    add_common_args(smoke_parser)
    
    return parser.parse_args()

def main() -> int:
    """Main function."""
    args: argparse.Namespace = parse_args()
    
    if not args.command:
        print("Error: No command specified. Use -h for help.")
        return 1
    
    # Set up logging
    log_file = None
    if args.command in ['train', 'eval']:
        ensure_dir(args.output_dir)
        log_file = os.path.join(args.output_dir, f'{args.command}.log')
    
    setup_logging(log_file)
    
    # Set device and seed
    device = get_device(args.device)
    set_seed(args.seed)
    
    try:
        if args.command == 'smoke':
            smoke_test()
        elif args.command == 'train':
            logger.info("Training command is not fully implemented yet.")
            logger.info("This implementation provides the complete model architecture ")
            logger.info("and all necessary components for training.")
            logger.info("To complete: implement Trainer class and training loop.")
        elif args.command == 'eval':
            logger.info("Evaluation command is not fully implemented yet.")
            logger.info("Use evaluate_model() function with your trained model.")
        elif args.command == 'predict':
            logger.info("Prediction command is not fully implemented yet.")
            logger.info("Use evaluate_model() function and extract predictions.")
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
    except Exception as e:
        logger.error(f"Error in {args.command}: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
