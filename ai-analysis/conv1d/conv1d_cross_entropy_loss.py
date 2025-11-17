"""CrossEntropy Loss Components for Conv1D

This module provides classification components for training Conv1D models
with CrossEntropyLoss instead of regression loss.

Reuses BLSTM CE infrastructure:
- Score ↔ class index mapping utilities from blstm_cross_entropy_loss
- Conv1DClassifier: Classification head variant of Conv1DRegressor
- 6-class output for C1 scores {0, 40, 80, 120, 160, 200}
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Import Conv1D base components
from conv1d import ModelConfig, masked_avgpool_1d, masked_maxpool_1d

# Import BLSTM CE mapping utilities
sys.path.append(str(Path(__file__).parent.parent / "blstm"))
from blstm_cross_entropy_loss import (
    NUM_CLASSES,
    scores_to_class_indices,
    class_indices_to_scores,
    logits_to_scores,
    validate_scores_for_ce,
)

__all__ = [
    "Conv1DClassifier",
    "NUM_CLASSES",
    "scores_to_class_indices",
    "class_indices_to_scores",
    "logits_to_scores",
    "validate_scores_for_ce",
]


class Conv1DClassifier(nn.Module):
    """1D Convolutional Neural Network for essay C1 score classification with CrossEntropyLoss.
    
    Mirrors Conv1DRegressor encoder but replaces regression head with
    6-way classification head for {0, 40, 80, 120, 160, 200}.
    
    Forward returns raw logits [batch_size, 6] for CE loss.
    Use logits_to_scores() to convert predictions to C1 scores.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Build convolutional layers dynamically (identical to Conv1DRegressor)
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
        
        # Dense head for classification (6 classes)
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, config.dense_neurons),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dense_neurons, NUM_CLASSES),
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
            logits: [batch_size, 6] raw logits for CrossEntropyLoss
        """
        # Handle different input formats (identical to Conv1DRegressor)
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
        
        # Classification head
        logits = self.head(x)  # [batch_size, NUM_CLASSES]
        
        return logits
    
    def predict_scores(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Forward + convert logits to scores.
        
        Returns:
            Predicted C1 scores [batch_size] in {0, 40, 80, 120, 160, 200}
        """
        logits = self.forward(x, lengths)
        return logits_to_scores(logits)
