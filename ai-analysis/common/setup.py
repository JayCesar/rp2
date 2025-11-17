"""Setup utilities for logging and performance optimizations

Provides functions for configuring logging and enabling GPU optimizations.
"""

import logging
import sys

import torch


def setup_logging(
    level: int = logging.INFO,
    format_string: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
) -> logging.Logger:
    """Configure logging with consistent formatting.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Log message format
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logging()
        >>> logger.info("Training started")
    """
    # Clear any existing handlers
    logging.root.handlers = []
    
    # Configure basic logging
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Get logger
    logger = logging.getLogger()
    
    # Reduce noise from some libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    return logger


def configure_cuda_optimizations(device: torch.device) -> None:
    """Enable CUDA performance optimizations.
    
    Enables:
    - TF32 for faster matmul on Ampere+ GPUs
    - CuDNN benchmarking for optimal conv algorithms
    
    Only applies optimizations if device is CUDA.
    
    Args:
        device: torch.device to check if CUDA
        
    Example:
        >>> device = torch.device("cuda")
        >>> configure_cuda_optimizations(device)
        # TF32 and cudnn benchmark now enabled
    """
    if device.type != "cuda":
        return
    
    # Enable TF32 for faster matmul on Ampere+ GPUs
    # TF32 provides ~8x speedup for matmul with minimal accuracy loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cudnn benchmarking for optimal conv algorithms
    # This finds the fastest conv algorithm for your specific hardware
    torch.backends.cudnn.benchmark = True
    
    logger = logging.getLogger(__name__)
    logger.info("CUDA optimizations enabled (TF32, cudnn benchmark)")


def get_optimal_workers(device: torch.device, default_gpu: int = 4, default_cpu: int = 2) -> int:
    """Get optimal number of DataLoader workers based on device.
    
    Args:
        device: Computation device
        default_gpu: Workers for GPU (default: 4)
        default_cpu: Workers for CPU (default: 2)
        
    Returns:
        Optimal number of workers
        
    Example:
        >>> device = get_device()
        >>> num_workers = get_optimal_workers(device)
        >>> DataLoader(dataset, num_workers=num_workers)
    """
    return default_gpu if device.type == "cuda" else default_cpu
