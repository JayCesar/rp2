"""Class frequency utilities shared across models.

Provides helpers to compute per-class frequencies and inverse-frequency
alpha weights for focal loss.

These functions are intentionally independent of any specific model
(Conv1D, BLSTM, etc.) and operate purely on C1 score labels.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import polars as pl
import torch
from torch.utils.data import DataLoader, Subset


def _scores_to_class_indices(scores: torch.Tensor, num_classes: int = 6) -> torch.Tensor:
    """Map C1 scores {0,40,80,120,160,200} to class indices {0..5}.

    This mirrors :func:`scores_to_class_indices` in the BLSTM CE module but is
    implemented locally to keep this helper free of model-specific imports.
    """

    class_indices = (scores / 40).long()
    if num_classes is not None:
        class_indices = class_indices.clamp(min=0, max=num_classes - 1)
    return class_indices


def calculate_alpha_from_frequency(class_frequencies: Sequence[int]) -> torch.Tensor:
    """Calculate alpha weights from class frequencies using inverse frequency.

    Args:
        class_frequencies:
            Iterable of sample counts per class.

    Returns:
        Alpha weights as a 1D tensor, clipped into ``[0.1, 0.9]``.
    """

    total = float(sum(class_frequencies))
    if total <= 0:
        # Degenerate case: no samples. Fall back to uniform weights.
        return torch.full((len(class_frequencies),), 1.0 / max(1, len(class_frequencies)))

    alphas: list[float] = []
    for freq in class_frequencies:
        alpha = 1.0 - (float(freq) / total)
        alpha = max(0.1, min(0.9, alpha))
        alphas.append(alpha)
    return torch.tensor(alphas, dtype=torch.float32)


def get_class_frequencies(dataset_or_df, num_classes: int = 6) -> list[int]:
    """Calculate class frequencies from a DataFrame or Dataset with C1 scores.

    Args:
        dataset_or_df:
            - A dataset/Subset with a ``.data['c1']`` column (Polars Series), or
            - A Polars DataFrame with a ``'c1'`` column.
        num_classes:
            Number of classes (default: 6).

    Returns:
        List[int]: sample counts per class.
    """

    # Handle Subset objects
    if isinstance(dataset_or_df, Subset):
        dataset = dataset_or_df.dataset  # type: ignore[assignment]
        indices = dataset_or_df.indices
        scores = torch.tensor(
            [dataset.data["c1"][i] for i in indices], dtype=torch.float32
        )
    # Handle Dataset-like objects with a ``data`` attribute
    elif hasattr(dataset_or_df, "data"):
        scores = torch.tensor(
            dataset_or_df.data["c1"].to_numpy(), dtype=torch.float32
        )
    # Handle Polars DataFrame directly
    else:
        if not isinstance(dataset_or_df, pl.DataFrame):  # type: ignore[unreachable]
            raise TypeError(
                "get_class_frequencies expects a Dataset/Subset with a 'data' attribute "
                "or a Polars DataFrame with a 'c1' column."
            )
        scores = torch.tensor(dataset_or_df["c1"].to_numpy(), dtype=torch.float32)

    class_indices = _scores_to_class_indices(scores, num_classes=num_classes).long()

    frequencies = [0] * num_classes
    for idx in class_indices:
        i = int(idx.item())
        if 0 <= i < num_classes:
            frequencies[i] += 1
    return frequencies


def get_class_frequencies_from_loader(
    loader: DataLoader, num_classes: int = 6
) -> list[int]:
    """Calculate class frequencies from a DataLoader.

    Args:
        loader:
            DataLoader yielding batches with a ``'targets'`` key containing C1 scores.
        num_classes:
            Number of classes (default: 6).

    Returns:
        List[int]: sample counts per class.
    """

    frequencies = [0] * num_classes

    for batch in loader:
        targets = batch["targets"]
        scores = targets.to(torch.float32)
        class_indices = _scores_to_class_indices(scores, num_classes=num_classes).long()

        for idx in class_indices:
            i = int(idx.item())
            if 0 <= i < num_classes:
                frequencies[i] += 1

    return frequencies


def print_class_distribution(frequencies: Sequence[int]) -> None:
    """Print class distribution statistics for C1 score classes.

    Args:
        frequencies:
            Iterable of sample counts per class, assumed to correspond to
            scores ``[0, 40, 80, 120, 160, 200]`` when length is 6.
    """

    total = float(sum(frequencies))
    scores = [0, 40, 80, 120, 160, 200]

    print("Class Distribution:")
    print("-" * 50)
    for i, freq in enumerate(frequencies):
        score = scores[i] if i < len(scores) else i
        pct = (freq / total) * 100 if total > 0 else 0.0
        print(f"Class {i} (score={score:3d}): {freq:6d} samples ({pct:5.2f}%)")
    print("-" * 50)
    print(f"Total: {int(total)} samples")


__all__ = [
    "calculate_alpha_from_frequency",
    "get_class_frequencies",
    "get_class_frequencies_from_loader",
    "print_class_distribution",
]