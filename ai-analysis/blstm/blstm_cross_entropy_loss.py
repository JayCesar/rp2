"""CrossEntropy Loss Components for BiLSTM

This module provides classification components for training BiLSTM models
with CrossEntropyLoss instead of regression loss.

Key Components:
- Score ↔ class index mapping utilities for C1 scores {0, 40, 80, 120, 160, 200}
- BiLSTMClassifier: Classification head variant of BiLSTMRegressor
- Helper functions for converting between score and class representations
"""

import torch
import torch.nn as nn
from typing import Union

# Score-to-class mapping for C1 scores
# Scores: {0, 40, 80, 120, 160, 200} map to classes: {0, 1, 2, 3, 4, 5}
SCORE_TO_CLASS_MAP = {
    0: 0,
    40: 1,
    80: 2,
    120: 3,
    160: 4,
    200: 5,
}

CLASS_TO_SCORE_MAP = {
    0: 0,
    1: 40,
    2: 80,
    3: 120,
    4: 160,
    5: 200,
}

VALID_SCORES = frozenset([0, 40, 80, 120, 160, 200])
NUM_CLASSES = 6


# Scalar mapping functions
def score_to_class_idx(score: int) -> int:
    """Convert C1 score to class index.
    
    Args:
        score: C1 score in {0, 40, 80, 120, 160, 200}
        
    Returns:
        Class index in {0, 1, 2, 3, 4, 5}
        
    Raises:
        ValueError: If score is not in valid set
        
    Examples:
        >>> score_to_class_idx(0)
        0
        >>> score_to_class_idx(120)
        3
        >>> score_to_class_idx(200)
        5
    """
    if score not in VALID_SCORES:
        raise ValueError(
            f"Invalid score {score}. Must be one of {sorted(VALID_SCORES)}"
        )
    return SCORE_TO_CLASS_MAP[score]


def class_idx_to_score(idx: int) -> int:
    """Convert class index to C1 score.
    
    Args:
        idx: Class index in {0, 1, 2, 3, 4, 5}
        
    Returns:
        C1 score in {0, 40, 80, 120, 160, 200}
        
    Raises:
        ValueError: If idx is not in valid range [0, 5]
        
    Examples:
        >>> class_idx_to_score(0)
        0
        >>> class_idx_to_score(3)
        120
        >>> class_idx_to_score(5)
        200
    """
    if idx not in range(NUM_CLASSES):
        raise ValueError(
            f"Invalid class index {idx}. Must be in range [0, {NUM_CLASSES-1}]"
        )
    return CLASS_TO_SCORE_MAP[idx]


# Vectorized tensor mapping functions
def scores_to_class_indices(scores: torch.Tensor) -> torch.Tensor:
    """Convert tensor of C1 scores to class indices.
    
    Args:
        scores: Tensor of C1 scores (any shape), values in {0, 40, 80, 120, 160, 200}
        
    Returns:
        Tensor of class indices (same shape as input), dtype int64
        
    Raises:
        ValueError: If any score is not in valid set
        
    Examples:
        >>> scores = torch.tensor([0, 40, 120, 200])
        >>> scores_to_class_indices(scores)
        tensor([0, 1, 3, 5])
    """
    # Validate all scores are in valid set
    scores_np = scores.cpu().numpy() if scores.is_cuda else scores.numpy()
    invalid = set(scores_np.flatten()) - VALID_SCORES
    if invalid:
        raise ValueError(
            f"Invalid scores found: {sorted(invalid)}. "
            f"All scores must be in {sorted(VALID_SCORES)}"
        )
    
    # Create mapping tensor for efficient vectorized conversion
    # Map score directly to class via division: score / 40 = class_idx
    class_indices = (scores / 40).long()
    
    return class_indices


def class_indices_to_scores(indices: torch.Tensor) -> torch.Tensor:
    """Convert tensor of class indices to C1 scores.
    
    Args:
        indices: Tensor of class indices (any shape), values in {0, 1, 2, 3, 4, 5}
        
    Returns:
        Tensor of C1 scores (same shape as input), dtype int64
        
    Raises:
        ValueError: If any index is not in valid range [0, 5]
        
    Examples:
        >>> indices = torch.tensor([0, 1, 3, 5])
        >>> class_indices_to_scores(indices)
        tensor([0, 40, 120, 200])
    """
    # Validate all indices are in valid range
    indices_np = indices.cpu().numpy() if indices.is_cuda else indices.numpy()
    invalid = [int(idx) for idx in indices_np.flatten() if idx < 0 or idx >= NUM_CLASSES]
    if invalid:
        raise ValueError(
            f"Invalid class indices found: {sorted(set(invalid))}. "
            f"All indices must be in range [0, {NUM_CLASSES-1}]"
        )
    
    # Convert class indices to scores: class_idx * 40 = score
    scores = indices * 40
    
    return scores.long()


def logits_to_scores(logits: torch.Tensor) -> torch.Tensor:
    """Convert classification logits to C1 scores via argmax.
    
    This is the inference function: takes raw logits from BiLSTMClassifier,
    applies argmax to get predicted class, then maps to C1 score.
    
    Args:
        logits: Raw logits from model with shape [..., num_classes]
        
    Returns:
        Tensor of predicted C1 scores, shape [...], dtype int64
        
    Examples:
        >>> logits = torch.randn(4, 6)  # batch_size=4, num_classes=6
        >>> scores = logits_to_scores(logits)
        >>> scores.shape
        torch.Size([4])
        >>> set(scores.tolist()).issubset({0, 40, 80, 120, 160, 200})
        True
    """
    # Get predicted class indices via argmax
    class_preds = logits.argmax(dim=-1)
    
    # Convert to scores
    scores = class_indices_to_scores(class_preds)
    
    return scores


def validate_scores_for_ce(scores: Union[torch.Tensor, list, tuple]) -> None:
    """Validate that all scores are in the valid set for CE training.
    
    Utility function to check data before training. Raises informative error
    if any out-of-domain scores are found.
    
    Args:
        scores: Scores to validate (tensor, list, or tuple)
        
    Raises:
        ValueError: If any score is not in {0, 40, 80, 120, 160, 200}
    """
    if isinstance(scores, torch.Tensor):
        scores_set = set(scores.cpu().numpy().flatten().tolist())
    else:
        scores_set = set(scores)
    
    invalid = scores_set - VALID_SCORES
    if invalid:
        raise ValueError(
            f"Found invalid C1 scores for CE training: {sorted(invalid)}. "
            f"All scores must be in {sorted(VALID_SCORES)}. "
            f"Check your dataset and ensure scores are properly rounded/snapped."
        )


# BiLSTMClassifier - Classification head variant
class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM for essay C1 score classification with CrossEntropyLoss.
    
    Mirrors BiLSTMRegressor encoder but replaces regression head with
    6-way classification head for {0, 40, 80, 120, 160, 200}.
    
    Forward returns raw logits [batch_size, 6] for CE loss.
    Use logits_to_scores() to convert predictions to C1 scores.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Import common attention aggregation
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from common import AttentionAggregation
        
        # Token projection
        if self.config.token_proj_dim and self.config.token_proj_dim > 0:
            self.token_proj = nn.Sequential(
                nn.LayerNorm(self.config.input_dim),
                nn.Linear(self.config.input_dim, self.config.token_proj_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            )
            lstm_input_dim = self.config.token_proj_dim
        else:
            self.token_proj = nn.Identity()
            lstm_input_dim = self.config.input_dim
        
        # LSTM layers
        hs1, hs2, hs3 = self.config.hidden_sizes[:3]
        direction_multiplier = 2
        
        self.lstm1 = nn.LSTM(
            input_size=lstm_input_dim,
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
        
        self.dropout = nn.Dropout(self.config.dropout)
        lstm_output_size = hs3 * direction_multiplier
        
        # Aggregation
        if config.aggregation == "attn":
            self.aggregation = AttentionAggregation(lstm_output_size)
        else:
            self.aggregation = None
        
        # Pre-head norm
        self.pre_head_norm = (
            nn.LayerNorm(lstm_output_size)
            if self.config.use_layer_norm
            else nn.Identity()
        )
        
        # Classification head (6 classes)
        if self.config.mlp_hidden:
            self.head = nn.Sequential(
                nn.Linear(lstm_output_size, self.config.mlp_hidden),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.mlp_hidden, NUM_CLASSES),
            )
        else:
            self.head = nn.Linear(lstm_output_size, NUM_CLASSES)
    
    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            tokens: [batch_size, max_seq_len, input_dim]
            lengths: [batch_size]
            
        Returns:
            logits: [batch_size, 6] raw logits for CrossEntropyLoss
        """
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        
        batch_size = tokens.shape[0]
        tokens = self.token_proj(tokens)
        
        # Pack and process through LSTM layers
        packed = pack_padded_sequence(
            tokens, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
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
        hidden, cell = hidden3, cell3
        
        # Aggregate
        if self.config.aggregation == "last":
            representation = torch.cat([hidden[-2], hidden[-1]], dim=1)
        elif self.config.aggregation == "mean":
            mask = torch.arange(lstm_output.shape[1], device=tokens.device).unsqueeze(
                0
            ).expand(batch_size, -1) < lengths.unsqueeze(1)
            masked_output = lstm_output * mask.unsqueeze(-1)
            representation = masked_output.sum(dim=1) / lengths.unsqueeze(-1).float()
        elif self.config.aggregation == "max":
            mask = torch.arange(lstm_output.shape[1], device=tokens.device).unsqueeze(
                0
            ).expand(batch_size, -1) < lengths.unsqueeze(1)
            masked_output = lstm_output.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            representation = masked_output.max(dim=1)[0]
        elif self.config.aggregation == "attn":
            representation = self.aggregation(lstm_output, lengths)
        else:
            raise ValueError(f"Unknown aggregation: {self.config.aggregation}")
        
        representation = self.pre_head_norm(representation)
        logits = self.head(representation)
        return logits
    
    def predict_scores(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward + convert logits to scores.
        
        Returns:
            Predicted C1 scores [batch_size] in {0, 40, 80, 120, 160, 200}
        """
        logits = self.forward(tokens, lengths)
        return logits_to_scores(logits)
