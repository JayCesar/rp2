"""Data utilities for essay datasets and data loading.

Provides:
- EssayDataset: Unified dataset class supporting token embeddings and features
- collate_batch: Batch collation with sequence padding
- create_data_loader: DataLoader factory with sensible defaults
- split_dataset: Train/val/test splitting with stratification support
"""

import re
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset, random_split


class EssayDataset(Dataset):
    """Dataset class for essay data supporting multiple input modes.
    
    Supports:
    - Token embeddings: 'essay_token_embeddings' column with [seq_len, 768] tensors
    - Feature vectors: SCREAMING_SNAKE_CASE feature columns
    - Vector mode (legacy): 'essay_vector' column with [768] vectors
    """

    def __init__(self, data: pl.DataFrame):
        """Initialize dataset from Polars DataFrame.
        
        Args:
            data: DataFrame with essay data. Must contain 'c1' (target) column.
                  Can contain 'essay_token_embeddings' for token mode, or
                  feature columns matching ^[A-Z0-9_]+$ pattern.
        """
        super().__init__()
        self.data = data
        cols = set(self.data.columns)
        
        # Detect input mode
        self.is_token_mode = "essay_token_embeddings" in cols
        self.is_vector_mode = "essay_vector" in cols
        
        # For feature mode, select SCREAMING_SNAKE_CASE columns
        if not (self.is_token_mode or self.is_vector_mode):
            snake_case_pattern = re.compile(r"^[A-Z0-9_]+$")
            self.feature_cols = [
                c for c in self.data.columns if c != "c1" and snake_case_pattern.match(c)
            ]
        else:
            self.feature_cols = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get single essay sample.
        
        Returns:
            dict with keys:
                - 'id': essay identifier
                - 'tokens': tensor [seq_len, input_dim] or [1, input_dim]
                - 'lengths': scalar tensor with sequence length
                - 'targets': scalar tensor with C1 score
        """
        row = self.data.row(idx, named=True)

        if self.is_token_mode:
            # Token embeddings mode: [seq_len, 768]
            token_embeddings = np.array(row["essay_token_embeddings"])
            tokens = torch.tensor(token_embeddings, dtype=torch.float32)
            # Use actual token length if available
            seq_length = int(row.get("essay_token_length", len(token_embeddings)))
            seq_length = max(1, min(seq_length, len(token_embeddings)))
            lengths = torch.tensor(seq_length, dtype=torch.long)
        elif self.is_vector_mode:
            # Vector mode (legacy): [1, 768]
            tokens = torch.tensor(row["essay_vector"], dtype=torch.float32).unsqueeze(0)
            lengths = torch.tensor(1, dtype=torch.long)
        else:
            # Feature mode: [1, num_features]
            if not self.feature_cols:
                raise KeyError(
                    "No feature columns found. Ensure dataset has SCREAMING_SNAKE_CASE columns besides 'c1'."
                )
            features = [float(row[c]) for c in self.feature_cols]
            tokens = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            lengths = torch.tensor(1, dtype=torch.long)

        return {
            "id": f"essay_{idx}",
            "tokens": tokens,
            "lengths": lengths,
            "targets": torch.tensor(row["c1"], dtype=torch.float32),
        }

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        """Batch getter for efficient DataLoader processing."""
        return [self.__getitem__(idx) for idx in indices]


def collate_batch(batch: list[dict[str, Any]], pad_value: float = 0.0) -> dict[str, Any]:
    """Collate function for batching variable-length sequences.
    
    Args:
        batch: List of samples from EssayDataset
        pad_value: Value to use for padding shorter sequences
        
    Returns:
        Batched dict with:
            - 'ids': list of essay IDs
            - 'tokens': padded tensor [batch_size, max_seq_len, input_dim]
            - 'lengths': tensor [batch_size] with actual sequence lengths
            - 'targets': tensor [batch_size] with C1 scores
    """
    ids = [item["id"] for item in batch]
    tokens = [item["tokens"] for item in batch]
    lengths = [item["lengths"] for item in batch]
    targets = [item["targets"] for item in batch]

    # Pad sequences to same length
    batched_tokens = pad_sequence(tokens, batch_first=True, padding_value=pad_value)
    batched_lengths = torch.stack(lengths)
    batched_targets = torch.stack(targets)

    return {
        "ids": ids,
        "tokens": batched_tokens,
        "lengths": batched_lengths,
        "targets": batched_targets,
    }


def create_data_loader(
    dataset: EssayDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = False,
) -> DataLoader:
    """Create DataLoader with sensible defaults for essay data.
    
    Args:
        dataset: EssayDataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        DataLoader configured for essay data
    """
    if num_workers > 0:
        try:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collate_batch,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,
            )
        except TypeError:
            # Fallback for older PyTorch versions without persistent_workers
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


def split_dataset(
    dataset: EssayDataset,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Split dataset into train/val/test subsets.
    
    Args:
        dataset: EssayDataset to split
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for test
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_subset, val_subset, test_subset)
    """
    total_size = len(dataset)
    val_size = int(val_ratio * total_size)
    test_size = int(test_ratio * total_size)
    train_size = total_size - val_size - test_size

    torch.manual_seed(seed)
    return random_split(dataset, [train_size, val_size, test_size])
