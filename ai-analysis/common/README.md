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

### `gamma_search.py` - FocalLoss Gamma Sweeps

You normally do **not** call `run_gamma_search` directly for Conv1D. Instead,
use the Conv1D CLI, which wires the correct factories and data pipeline.

#### Conv1D features CLI (recommended)

From the project root:

```bash
uv run python ai-analysis/conv1d/conv1d_train_on_features_focal_loss.py
```

This command **always** runs a FocalLoss gamma sweep; single-run training
(without a sweep) is not supported.

Optional flags:

- `--gamma-grid "0.5,1.0,2.0,3.5,5.0,10.0"` – custom gamma values (otherwise
  `DEFAULT_GAMMA_VALUES` is used).
- `--epochs-per-gamma 5` – override `TrainConfig.epochs` for each gamma.
- `--max-samples 2000` – cap training samples per gamma (val/test always use
  full data).

This command internally calls `run_gamma_search` with
`trainer_factory_conv1d_features` and `dataloaders_factory_features`, and
writes results under:

- `ai-analysis/conv1d/runs/features_focal_loss/conv1d_model/gamma_sweep/`

including `results_by_gamma.*`, `best_gamma.json`, and per-gamma metric
subdirectories.

#### BLSTM FocalLoss CLIs (features & vectorized essays)

BLSTM FocalLoss training scripts expose the same gamma-search behaviour as the
Conv1D ones, but using BiLSTM classifiers.

From the project root:

```bash
# Features-based BLSTM FocalLoss gamma sweep
uv run python ai-analysis/blstm/blstm_train_on_features_focal_loss.py \
  --max-samples 2000 \
  --epochs-per-gamma 5 \
  --gamma-grid "0.5,1,2,4,8"

# Vectorized-essay BLSTM FocalLoss gamma sweep
uv run python ai-analysis/blstm/blstm_train_on_vectorized_essays_focal_loss.py \
  --max-samples 2000 \
  --epochs-per-gamma 5 \
  --gamma-grid "0.5,1,2,4,8"
```

Notes:

- If `--gamma-grid` is omitted, both BLSTM scripts use the shared
  `DEFAULT_GAMMA_VALUES` grid.
- If `--epochs-per-gamma` is omitted, they fall back to `TrainConfig.epochs`.
- `--max-samples` caps **only** the training split; validation/test always use
  the full data.
- The vectorized-essays script expects
  `extended_essay-br_preprocessed_for_BLSTM.parquet`; if missing, it will try
  to fuse `*_part1.parquet` and `*_part2.parquet` automatically using
  `generated_datasets/fuse_blstm_parquet.py`.

Outputs:

- Features FocalLoss (BLSTM):
  - `ai-analysis/blstm/runs/features_focal_loss/blstm_model/gamma_sweep/`
- Vectorized essays FocalLoss (BLSTM):
  - `ai-analysis/blstm/runs/vectorized_essays_focal_loss/blstm_model/gamma_sweep/`

Each gamma subdirectory follows the same conventions as Conv1D:

- `metrics_best.*` for per-gamma metrics (including a `gamma` column).
- `training_history.*` for per-epoch history when available.
- `validation_predictions_best.*` with `id,target,pred,pred_snapped`.

#### Direct API (advanced / custom models)

```python
from pathlib import Path
from common import (
    DEFAULT_GAMMA_VALUES,
    metric_key_for_selection,
    run_gamma_search,
)

results_by_gamma, best_gamma, best_metrics = run_gamma_search(
    trainer_factory=trainer_factory,
    dataloaders_factory=dataloaders_factory,
    gamma_values=DEFAULT_GAMMA_VALUES,
    num_classes=6,
    output_root=Path("runs/my_model/gamma_sweep"),
    seed=42,
    alpha=None,
    max_samples=None,
)

for gamma in sorted(
    results_by_gamma,
    key=lambda g: metric_key_for_selection(results_by_gamma[g]),
):
    print(gamma, results_by_gamma[gamma]["qwk"], results_by_gamma[gamma]["mae"])

print("Best gamma:", best_gamma)
print("Best metrics:", best_metrics)
```

**Factory contracts**

- ``dataloaders_factory(max_samples: int | None) -> tuple[train_loader, val_loader, test_loader | None]``
- ``trainer_factory(out_dir: Path, train_loader, val_loader, test_loader | None) -> Trainer``

Gamma selection is performed by ``metric_key_for_selection(metrics)`` which
returns ``(-qwk, -kappa, mae)`` so that ``min()`` prefers higher QWK,
then higher Kappa, then lower MAE.

**Minimal end-to-end example (dummy trainer)**

```python
from pathlib import Path
from common import run_gamma_search, metric_key_for_selection

# 1) Minimal factories -------------------------------------------------

class DummyTrainer:
    def __init__(self, out_dir: Path, *_):
        self.out_dir = out_dir
        self.training_history = []
        self.best_val_predictions = []
        self.criterion = type("C", (), {"gamma": 0.0})()

    def train(self) -> dict[str, float]:
        gamma = float(getattr(self.criterion, "gamma", 0.0))
        return {"mae": abs(2.0 - gamma), "qwk": gamma, "kappa": gamma / 2.0}


def dataloaders_factory(max_samples: int | None):
    _ = max_samples
    return None, None, None


def trainer_factory(out_dir: Path, train_loader, val_loader, test_loader):
    _ = (train_loader, val_loader, test_loader)
    return DummyTrainer(out_dir)

# 2) Run the sweep -----------------------------------------------------

results_by_gamma, best_gamma, best_metrics = run_gamma_search(
    trainer_factory=trainer_factory,
    dataloaders_factory=dataloaders_factory,
    gamma_values=[0.5, 1.0, 2.0, 3.5, 5.0, 10.0],
    num_classes=3,
    output_root=Path("runs/dummy_gamma_sweep"),
    seed=123,
    alpha=None,
    max_samples=None,
)

for gamma in sorted(results_by_gamma, key=lambda g: metric_key_for_selection(results_by_gamma[g])):
    print(gamma, results_by_gamma[gamma])

print("Best gamma:", best_gamma)
print("Best metrics:", best_metrics)
```

**Output layout**

```
output_root/
  results_by_gamma.csv
  results_by_gamma.parquet
  best_gamma.json
  metrics_best_overall.csv     # one-row table for the selected gamma
  metrics_best_overall.parquet
  gamma_0_5/
    metrics_best.*             # includes a `gamma` column
    training_history.*         # if trainer.training_history is non-empty
    validation_predictions_*   # if trainer.best_val_predictions is non-empty
  gamma_1/
  ...
```

Each per-gamma directory matches the conventions used by Conv1D trainers:

- `metrics_best.*` contains a single-row metrics table and a `gamma` column.
- `validation_predictions_best.*` has the schema `id,target,pred,pred_snapped`.

The root-level `results_by_gamma.*` adds an `is_best_gamma` boolean column to
indicate which gamma was selected, and `metrics_best_overall.*` gives a
one-row summary (with `gamma`) for the best configuration.

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
