"""Helper to calculate class frequencies from dataset for focal loss alpha weights.

This module now delegates to :mod:`ai_analysis.common.class_frequencies` so
that Conv1D and BLSTM models share the same implementation.

Existing imports remain supported for backwards compatibility::

    from conv1d.calculate_class_frequencies import get_class_frequencies
"""

from __future__ import annotations

from ..common.class_frequencies import (
    calculate_alpha_from_frequency,
    get_class_frequencies,
    get_class_frequencies_from_loader,
    print_class_distribution,
)

__all__ = [
    "calculate_alpha_from_frequency",
    "get_class_frequencies",
    "get_class_frequencies_from_loader",
    "print_class_distribution",
]
