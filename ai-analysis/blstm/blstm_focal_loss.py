"""Focal-loss compatible BLSTM classification utilities.

This module re-exports the C1 score ↔ class index mappings and classifier
used for BLSTM CrossEntropy training so they can be shared with the
FocalLoss trainer without duplicating logic.
"""

from __future__ import annotations

# We intentionally import from the local CE module (rather than from common)
# to keep all BLSTM-specific score mapping logic in a single place.
from blstm_cross_entropy_loss import (
    NUM_CLASSES,
    BiLSTMClassifier,
    class_idx_to_score,
    class_indices_to_scores,
    logits_to_scores,
    score_to_class_idx,
    scores_to_class_indices,
    validate_scores_for_ce,
)

__all__ = [
    "NUM_CLASSES",
    "BiLSTMClassifier",
    "score_to_class_idx",
    "class_idx_to_score",
    "scores_to_class_indices",
    "class_indices_to_scores",
    "logits_to_scores",
    "validate_scores_for_ce",
]
