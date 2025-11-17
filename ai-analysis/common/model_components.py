"""Reusable model components for neural networks.

Provides:
- AttentionAggregation: Attention-based sequence pooling
"""

import torch
import torch.nn as nn


class AttentionAggregation(nn.Module):
    """Attention-based sequence aggregation.
    
    Computes weighted sum of sequence elements using learned attention weights.
    Properly handles variable-length sequences with padding masks.
    """

    def __init__(self, hidden_size: int):
        """Initialize attention module.
        
        Args:
            hidden_size: Dimensionality of input sequence elements
        """
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Apply attention aggregation to sequences.
        
        Args:
            sequences: Tensor of shape [batch_size, seq_len, hidden_size]
            lengths: Tensor of shape [batch_size] with actual sequence lengths
            
        Returns:
            Aggregated tensor of shape [batch_size, hidden_size]
        """
        batch_size, seq_len, hidden_size = sequences.shape

        # Compute attention weights: [batch_size, seq_len, 1]
        attention_weights = self.attention(sequences)

        # Create mask for padding positions
        mask = torch.arange(seq_len, device=sequences.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]

        # Apply mask to attention weights (set padding to -inf before softmax)
        attention_weights = attention_weights.masked_fill(~mask, float("-inf"))
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Weighted sum: [batch_size, hidden_size]
        aggregated = (sequences * attention_weights).sum(dim=1)

        return aggregated
