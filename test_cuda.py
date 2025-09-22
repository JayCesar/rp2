#!/usr/bin/env python3

import torch

print("=== PyTorch CUDA Test ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"GPU memory: {round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1)} GB")
    
    # Test tensor operations on GPU
    print("\n=== GPU Test ===")
    device = torch.device("cuda")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.matmul(x, y)
    print(f"Matrix multiplication on GPU successful!")
    print(f"Result tensor device: {z.device}")
    print(f"Result tensor shape: {z.shape}")
else:
    print("CUDA is not available. Please check your installation.")