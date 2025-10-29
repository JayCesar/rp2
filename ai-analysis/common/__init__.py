"""Common utilities shared across all AI models

This package contains reusable components for:
- Device management and seeding
- Data scaling and preprocessing  
- Metrics computation
- Model checkpointing
- Logging configuration
"""

from .device import get_device, set_seed
from .io_utils import ensure_dir, save_dataset
from .metrics import MetricsAccumulator, TargetScaler
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
]
