"""Common utilities shared across all AI models

This package contains reusable components for:
- Device management and seeding
- Data scaling and preprocessing  
- Metrics computation
- Model checkpointing
- Logging configuration
- Data loading and dataset utilities
"""

from .data_utils import (
    EssayDataset,
    collate_batch,
    create_data_loader,
    split_dataset,
)
from .device import get_device, set_seed
from .gamma_search import (
    DEFAULT_GAMMA_VALUES,
    metric_key_for_selection,
    run_gamma_search,
)
from .io_utils import ensure_dir, save_dataset
from .metrics import MetricsAccumulator, TargetScaler
from .model_components import AttentionAggregation
from .setup import configure_cuda_optimizations, setup_logging

__all__ = [
    # Device & reproducibility
    "get_device",
    "set_seed",
    "configure_cuda_optimizations",
    # I/O
    "ensure_dir",
    "save_dataset",
    # Metrics
    "MetricsAccumulator",
    "TargetScaler",
    # Setup
    "setup_logging",
    # Data utilities
    "EssayDataset",
    "collate_batch",
    "create_data_loader",
    "split_dataset",
    # Model components
    "AttentionAggregation",
    # Gamma search utilities
    "DEFAULT_GAMMA_VALUES",
    "metric_key_for_selection",
    "run_gamma_search",
]
