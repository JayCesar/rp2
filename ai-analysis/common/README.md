# Common Utilities Package

Shared utilities for all AI models in the project. This package provides reusable components extracted from `blstm.py` and `utils.py` for better code organization and maintainability.

## 📦 Modules

### `device.py` - Device Management & Reproducibility
```python
from common import get_device, set_seed, seed_worker

# Auto-detect best device
device = get_device("auto")  # Returns cuda/mps/cpu

# Set seed for reproducibility
set_seed(42)

# Use with DataLoader for deterministic workers
DataLoader(dataset, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(42))
```

### `io_utils.py` - File Operations & Dataset Management
```python
from common import ensure_dir, save_dataset, load_dataset

# Create directories
output_dir = ensure_dir("runs/experiment_1")

# Save dataset in multiple formats
save_dataset(df, "preprocessed_essays", "csv", "parquet", "json")

# Load with lazy evaluation
df = load_dataset("preprocessed_essays", extension="parquet", lazy=True)
```

### `metrics.py` - Evaluation Metrics & Scaling
```python
from common import MetricsAccumulator, TargetScaler, quadratic_weighted_kappa

# Accumulate metrics across batches
metrics = MetricsAccumulator()
for batch in data_loader:
    preds, targets = model(batch), batch["targets"]
    metrics.update(preds, targets, batch["ids"])

results = metrics.compute_metrics()
print(f"MAE: {results['mae']:.2f}, QWK: {results['qwk']:.3f}")

# Scale targets
scaler = TargetScaler("minmax")
scaler.fit(train_scores)
scaled = scaler.transform(test_scores)
```

###  `setup.py` - Logging & Performance Configuration
```python
from common import setup_logging, configure_cuda_optimizations, get_optimal_workers

# Configure logging
logger = setup_logging()

# Enable CUDA optimizations (TF32, cudnn benchmark)
configure_cuda_optimizations(device)

# Get optimal worker count
num_workers = get_optimal_workers(device)  # 4 for GPU, 2 for CPU
```

## 🎯 Benefits

### Before (Duplicated Code)
```python
# In blstm.py
def get_device(...): ...
class MetricsAccumulator: ...

# In utils.py
def ensure_dir(...): ...
def save_dataset(...): ...

# In conv1d.py - duplicated imports
sys.path.append(...)
from blstm import get_device, MetricsAccumulator
```

### After (Centralized Utilities)
```python
# In any module
from common import (
    get_device, 
    set_seed,
    MetricsAccumulator,
    TargetScaler,
    ensure_dir,
    save_dataset,
)

# Clean, maintainable, testable
```

## 📊 Extracted Functionality

| Original Location | New Location | Components |
|-------------------|--------------|------------|
| `blstm.py` | `common/device.py` | `get_device`, `set_seed` |
| `blstm.py` | `common/metrics.py` | `MetricsAccumulator`, `TargetScaler`, `quadratic_weighted_kappa` |
| `utils.py` | `common/io_utils.py` | `ensure_dir`, `save_dataset` |
| `utils.py` | `common/setup.py` | `setup_logging` |
| New | `common/setup.py` | `configure_cuda_optimizations`, `get_optimal_workers` |

## 🔧 Migration Guide

### For Existing Code

**Old import pattern:**
```python
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent / "blstm"))
from blstm import get_device, MetricsAccumulator
```

**New import pattern:**
```python
# If ai-analysis is in sys.path
from common import get_device, MetricsAccumulator

# Or with relative imports
from ..common import get_device, MetricsAccumulator
```

### For New Code

Always import from `common` package:
```python
from common import (
    # Device
    get_device,
    set_seed,
    # Metrics
    MetricsAccumulator,
    TargetScaler,
    # I/O
    ensure_dir,
    save_dataset,
    load_dataset,
    # Setup
    setup_logging,
    configure_cuda_optimizations,
)
```

## 🏗️ Architecture

```
ai-analysis/
├── common/              # Shared utilities (NEW!)
│   ├── __init__.py     # Package exports
│   ├── device.py       # Device & reproducibility
│   ├── io_utils.py     # File operations
│   ├── metrics.py      # Evaluation metrics
│   ├── setup.py        # Logging & optimizations
│   └── README.md       # This file
├── blstm/              # BiLSTM model
├── conv1d/             # Conv1D model
└── feature_extraction/ # Data preprocessing
```

## ✅ Quality Improvements

1. **No Code Duplication**: Single source of truth for shared utilities
2. **Better Testability**: Each module can be tested independently
3. **Clear Dependencies**: Explicit imports show what's used where
4. **Easy Maintenance**: Update once, use everywhere
5. **Type Safety**: Full type hints throughout
6. **Documentation**: Comprehensive docstrings with examples

## 📝 Design Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **No Side Effects**: Pure functions where possible
3. **Explicit Over Implicit**: Clear function names and parameters
4. **Backward Compatible**: Old imports still work (deprecation path)
5. **Well Documented**: Every public function has examples

## 🚀 Performance Features

- `configure_cuda_optimizations()`: Enables TF32 and cudnn benchmarking
- `get_optimal_workers()`: Platform-specific DataLoader configuration
- `load_dataset(lazy=True)`: Lazy evaluation for large datasets
- `MetricsAccumulator`: Memory-efficient streaming metrics computation

## 🔄 Future Enhancements

- [ ] Add caching decorators for expensive operations
- [ ] Implement distributed training utilities
- [ ] Add model checkpointing utilities
- [ ] Create visualization helpers
- [ ] Add profiling utilities

---

**Version**: 1.0  
**Created**: 2024-10-24  
**Last Updated**: 2024-10-24
