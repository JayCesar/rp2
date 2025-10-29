# Project Dependencies

Complete list of all packages used across the ai-analysis project.

## 📦 Core Dependencies

### Machine Learning & Deep Learning
- **torch** (PyTorch) - Neural network framework
  - Used in: `blstm/`, `conv1d/`, `BERT/`, `common/`
  - Features: Model training, GPU acceleration, autograd
  
- **transformers** (Hugging Face) - Pre-trained models
  - Used in: `BERT/`, `feature_extraction/`
  - Models: BERTimbau (neuralmind/bert-base-portuguese-cased)
  
- **sklearn** (scikit-learn) - Machine learning utilities
  - Used in: `blstm/`, `conv1d/`, `common/metrics.py`, `linear_regression/`
  - Features: Metrics (kappa, accuracy), train_test_split
  
- **scipy** - Scientific computing
  - Used in: `common/metrics.py`, `blstm/`
  - Features: Pearson correlation, statistical tests

### Data Processing
- **polars** - Fast DataFrame library
  - Used in: Throughout the project
  - Features: Lazy evaluation, fast CSV/Parquet operations
  
- **numpy** - Numerical computing
  - Used in: Throughout the project
  - Features: Array operations, mathematical functions
  
- **pandas** - Data manipulation (legacy)
  - Used in: Some older scripts
  - Note: Being replaced by Polars

### Natural Language Processing
- **spacy** - NLP toolkit
  - Used in: `feature_extraction/`
  - Model: `pt_core_news_md` (Portuguese medium model)
  - Features: Tokenization, lemmatization, POS tagging
  
- **language_tool_python** - Grammar checking
  - Used in: `feature_extraction/`
  - Features: Grammar error detection for Portuguese text

### Progress & Visualization
- **tqdm** - Progress bars
  - Used in: `blstm/trainer.py`, `conv1d/trainer.py`
  - Features: Training progress visualization
  
- **matplotlib** - Plotting library
  - Used in: Data analysis scripts
  - Features: Visualization of metrics and distributions

## 📚 Standard Library Modules

### System & I/O
- **pathlib** - Modern path handling
- **os** - Operating system interface
- **sys** - System-specific parameters
- **subprocess** - Process management
- **logging** - Logging facility
- **json** - JSON encoding/decoding
- **pickle** - Object serialization

### Data Structures & Algorithms
- **collections** - Specialized container datatypes
- **dataclasses** - Data classes for configurations
- **typing** - Type hints support
- **re** - Regular expressions
- **string** - String operations
- **argparse** - Command-line argument parsing

### Utilities
- **time** - Time access and conversions
- **random** - Random number generation

## 🔧 Development Tools

### Type Checking
- **pyright** - Static type checker
  - Config: `pyrightconfig.json`
  - Python version: 3.13

## 📋 Package Usage Matrix

| Package | Conv1D | BiLSTM | BERT | Features | Common |
|---------|--------|--------|------|----------|--------|
| torch | ✅ | ✅ | ✅ | ✅ | ✅ |
| polars | ✅ | ✅ | ✅ | ✅ | ✅ |
| numpy | ✅ | ✅ | ✅ | ✅ | ✅ |
| sklearn | ✅ | ✅ | ✅ | ✅ | ✅ |
| scipy | ✅ | ✅ | - | - | ✅ |
| transformers | - | - | ✅ | ✅ | - |
| spacy | - | - | - | ✅ | - |
| language_tool_python | - | - | - | ✅ | - |
| tqdm | ✅ | ✅ | ✅ | - | - |
| matplotlib | - | - | ✅ | ✅ | - |

## 🎯 Installation

### Using uv (Recommended)
```bash
uv sync
```

### Using pip
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers polars numpy scipy scikit-learn
pip install spacy language-tool-python tqdm matplotlib
python -m spacy download pt_core_news_md
```

## 📌 Version Requirements

- **Python**: 3.13+
- **PyTorch**: Latest stable (2.0+)
- **CUDA**: 12.1+ (for GPU support)
- **Transformers**: 4.30+
- **Polars**: 0.19+

## 🔍 Import Paths Configured

The `pyrightconfig.json` includes the following paths for proper type checking:

```json
{
  "extraPaths": [
    "ai-analysis",
    "ai-analysis/common",
    "ai-analysis/blstm",
    "ai-analysis/conv1d",
    "ai-analysis/BERT",
    "ai-analysis/feature_extraction"
  ]
}
```

This allows clean imports like:
```python
from common import get_device, MetricsAccumulator
from conv1d import Conv1DRegressor
from blstm import BiLSTMRegressor
```

## 🚀 Performance Dependencies

### GPU Acceleration
- **CUDA Toolkit** - For GPU training
- **cuDNN** - Optimized primitives for deep neural networks

### Optimizations Enabled
- TF32 (Tensor Float 32) for Ampere+ GPUs
- cuDNN benchmarking for optimal convolution algorithms
- Mixed precision training (AMP)
- Pin memory for faster GPU transfers

## 📝 Notes

1. **Polars over Pandas**: We prefer Polars for its speed and lazy evaluation
2. **BERTimbau**: Portuguese BERT model from neuralmind
3. **spaCy Model**: Requires separate download: `python -m spacy download pt_core_news_md`
4. **Type Stubs**: Some packages may show warnings about missing type stubs
5. **GPU Support**: Install PyTorch with CUDA support for GPU acceleration

## 🔄 Migration from Legacy

Some older scripts still use:
- `pandas` → Migrating to `polars`
- Manual path manipulation → Using proper package structure
- `sys.path.append()` → Using relative imports

## 📚 Additional Resources

- PyTorch Documentation: https://pytorch.org/docs/
- Transformers Documentation: https://huggingface.co/docs/transformers/
- Polars Documentation: https://pola-rs.github.io/polars/
- spaCy Documentation: https://spacy.io/usage
- BERTimbau Model: https://huggingface.co/neuralmind/bert-base-portuguese-cased

---

**Last Updated**: 2024-10-24  
**Maintained by**: Project Team
