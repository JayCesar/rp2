"""Device management and reproducibility utilities

Provides functions for device selection and deterministic seeding across all frameworks.
"""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    """Auto-detect or select the best available device.
    
    Args:
        preference: Device preference - 'auto', 'cpu', 'cuda', or 'mps'
        
    Returns:
        torch.device for computation
        
    Example:
        >>> device = get_device("auto")
        >>> model.to(device)
    """
    if preference != "auto":
        return torch.device(preference)

    # Check for CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        return device

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
        return device

    # Fallback to CPU
    device = torch.device("cpu")
    logger.info("Using CPU")
    return device


def set_seed(seed: int) -> None:
    """Set seeds for reproducible results across all frameworks.
    
    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CuDNN (makes it deterministic)
    
    Args:
        seed: Random seed to use
        
    Example:
        >>> set_seed(42)
        >>> # All random operations are now deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def seed_worker(worker_id: int) -> None:
    """Seed worker for deterministic DataLoader with multiple workers.
    
    Use as worker_init_fn in DataLoader for reproducibility.
    
    Args:
        worker_id: Worker ID (automatically provided by DataLoader)
        
    Example:
        >>> DataLoader(dataset, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(42))
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
