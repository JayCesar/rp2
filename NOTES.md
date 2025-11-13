# Implementation Notes

## CrossEntropy Loss Implementation for BiLSTM

### Decision: Option B - True Classification Head

**Date**: 2025-11-08  
**Context7 MCP Status**: Unavailable at implementation time

**Decision Made**: Proceed with **Option B** - True classification head with 6 logits for CrossEntropyLoss.

#### Rationale

Based on standard PyTorch best practices for classification:

1. **Standard API**: `nn.CrossEntropyLoss` expects:
   - Input: Raw logits with shape `[batch_size, num_classes]` (no softmax/log_softmax)
   - Target: Class indices with shape `[batch_size]` and dtype `torch.long` (int64)

2. **Classification Head Design**:
   - Classification head outputs 6 logits corresponding to classes: {0, 40, 80, 120, 160, 200}
   - Model learns class boundaries naturally through backprop on CE loss
   - Maintains probabilistic interpretation (logits → softmax → class probabilities)

3. **Why NOT Option A** (Regressor + Snapping):
   - Snapped outputs are not logits - they're discrete values
   - Would require surrogate loss formulations (e.g., distance-based logits, soft labeling)
   - Non-standard approach with unclear training dynamics
   - Loses gradient information at bucket boundaries

#### Class Mapping

Score-to-class mapping for C1 essay scores:
- Score 0 → Class 0
- Score 40 → Class 1
- Score 80 → Class 2
- Score 120 → Class 3
- Score 160 → Class 4
- Score 200 → Class 5

#### Implementation Plan

1. **Core Components**:
   - `BiLSTMClassifier`: Reuses BiLSTMRegressor encoder, replaces head with `nn.Linear(hidden_dim, 6)`
   - Score ↔ class index mapping utilities (vectorized for torch.Tensor)
   - CE-specific training loop: `blstm_training_ce.py`

2. **Training Scripts**:
   - `blstm_train_on_features_ce.py` → outputs to `runs/features_ce/`
   - `blstm_train_on_vectorized_essays_ce.py` → outputs to `runs/vectorized_essays_ce/`

3. **Metrics Preservation**:
   - Convert logits → argmax → class idx → score for evaluation
   - Compute all existing metrics (MAE, RMSE, QWK, Kappa, Pearson, step_accuracy) on score scale
   - No changes to metric computation logic

4. **Configuration Reuse**:
   - `ModelConfig`, `TrainConfig`, `SerializationConfig` remain unchanged
   - CE scripts accept same configs as regression path
   - Metadata includes `criterion="CrossEntropyLoss"`, `num_classes=6`

5. **Isolation**:
   - All CE code is additive - no breaking changes to regression system
   - Separate run directories prevent output collision
   - Clear "ce" labeling in filenames, run names, and logs

#### References

- PyTorch CrossEntropyLoss docs: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Standard practice: Classification head outputs raw logits; loss combines LogSoftmax + NLLLoss internally

---

## Code Map (Updated as implementation progresses)

### Existing BLSTM System - Detailed Audit

#### Core Model & Configs (`ai-analysis/blstm/blstm.py`)
- **BiLSTMRegressor**: 3-layer bidirectional LSTM with configurable aggregation
  - Encoder: token_proj → lstm1/2/3 (with dropout between layers) → aggregation
  - Aggregation modes: "last", "mean", "max", "attn" (uses AttentionAggregation)
  - Head: optional MLP + final Linear(hidden_dim, 1) for regression
  - Forward returns: [batch_size] predictions
  - Method: `predict_and_optionally_clamp()` for clamping during eval
- **ModelConfig**: `hidden_sizes`, `input_dim`, `num_layers`, `dropout`, `aggregation`, `mlp_hidden`, `use_layer_norm`, `token_proj_dim`, `output_range`
- **TrainConfig**: `epochs`, `batch_size`, `lr`, `weight_decay`, `optimizer`, `scheduler`, `plateau_*`, `grad_clip_norm`, `early_stopping_patience`, `seed`, `device`, `use_amp`, `amp_dtype`, `target_scaler`
- **SerializationConfig**: `output_dir`, `save_best_only`, `keep_last_k`
- **ScoreConstants**: `MIN=0`, `MAX=200`, `STEP=40`
- **Utilities**: `snap_to_step()`, `round_to_c1_levels()`, `quadratic_weighted_kappa()`, `get_device()`, `set_seed()`, `ensure_dir()`
- **MetricsAccumulator**: Computes MAE, RMSE, R², Kappa, QWK, Pearson, step_accuracy, mae_step
- **TargetScaler**: Modes "none", "minmax", "standard" for target scaling

#### Trainer (`ai-analysis/blstm/trainer.py`)
- **BiLSTMTrainer**: Complete training loop with AMP, schedulers, early stopping
  - `_train_epoch()`: Forward, loss (MAE default), backward, grad clip, optimizer step
  - `_validate()`: Eval mode, compute metrics via MetricsAccumulator
  - `train()`: Epoch loop, best model tracking (by val_mae), checkpointing
  - Schedulers: ReduceLROnPlateau, OneCycleLR, or none
  - Optimizer: AdamW (fused if CUDA available)
  - AMP: torch.autocast + GradScaler
  - State: `training_history`, `best_val_predictions`

#### Training Scripts
- **`blstm_training.py`**: Main orchestration
  - `create_component1_config()`: Creates ModelConfig for "vectorized_essays" or "features"
  - `train_component1_standard()`: Splits data, creates loaders, trainer, trains, saves
  - `train_on_vectorized_essays()`: Loads parquet, creates EssayDataset, calls train
  - `train_on_features()`: Same for features dataset
  - Output dirs: `runs/vectorized_essays/blstm_model/`, `runs/features/blstm_model/`
- **`blstm_train_on_features.py`**: Script wrapper for features (if exists - to verify)
- **`blstm_train_on_vectorized_essays.py`**: Script wrapper for essays (if exists - to verify)

#### Shared Utilities (`ai-analysis/common/`)
- **`dataset.py`**: `EssayDataset` - Polars DataFrame wrapper
- **`data_utils.py`**: `collate_batch()`, `create_data_loader()`, `split_dataset()`
- **`attention.py`**: `AttentionAggregation` - Attention pooling over sequences
- **`metrics.py`**: `MetricsAccumulator`, `TargetScaler`
- **`device.py`**: `get_device()`, `set_seed()`
- **`io_utils.py`**: `ensure_dir()`, `save_dataset()`
- **`evaluation.py`**: `save_metrics()`, `save_validation_predictions()`, `format_metrics_log()`

#### Key Patterns to Mirror
1. **Encoder Reuse**: BiLSTMClassifier should reuse lstm1/2/3 structure exactly
2. **Config Compatibility**: CE trainer must accept same ModelConfig, TrainConfig, SerializationConfig
3. **Metrics in Score Space**: Always convert predictions back to {0,40,80,120,160,200} before metrics
4. **AMP Pattern**: `torch.autocast("cuda", enabled=use_amp)` in forward; GradScaler for backward
5. **Checkpointing**: Save model_state_dict, optimizer, scheduler, scaler, config, metrics, epoch
6. **Best Tracking**: Use `best_val_mae` (or same metric as regression) for checkpoint selection
7. **Dataset Shape**: EssayDataset returns dict with "id", "embedding" (or features), "score"

### New CE System (✅ Created)

#### BiLSTM CrossEntropy Implementation

**Files Created**:
- ✅ `ai-analysis/blstm/blstm_cross_entropy_loss.py` - BiLSTMClassifier + mapping utilities (374 lines)
- ✅ `ai-analysis/blstm/trainer_cross_entropy_loss.py` - BiLSTMCETrainer class (491 lines)
- ✅ `ai-analysis/blstm/blstm_train_on_features_cross_entropy_loss.py` - Features CE script (349 lines)
- ✅ `ai-analysis/blstm/blstm_train_on_vectorized_essays_cross_entropy_loss.py` - Essays CE script (356 lines)
- ✅ `tests/test_cross_entropy_loss_mapping.py` - Mapping tests (23 tests, all passing)
- ✅ `tests/test_bilstm_cross_entropy_loss_classifier.py` - Classifier tests (19 tests, all passing)

**Status**: ✅ **COMPLETE** - All components implemented and tested

**Test Results**:
- ✅ 23/23 mapping tests passed
- ✅ 19/19 classifier tests passed
- ✅ 2/2 integration smoke tests passed
- ✅ Feature mode training verified (1 epoch on 500 samples)
- ✅ Predictions confirmed to be valid scores from {0, 40, 80, 120, 160, 200}
- ✅ Checkpoints include `loss_type: "CrossEntropyLoss"` metadata

**Output Directories**:
- Features CE: `runs/features_cross_entropy_loss/blstm_model/`
- Essays CE: `runs/vectorized_essays_cross_entropy_loss/blstm_model/`

**Key Implementation Details**:
1. **Feature Mode Handling**: Trainer automatically reshapes [B, F] → [B, 1, F] with lengths=[1]*B
2. **Metrics**: All computed in score space after logits→argmax→class→score conversion
3. **Best Tracking**: Uses best_val_mae like regression system
4. **Configs**: Full compatibility with existing ModelConfig/TrainConfig/SerializationConfig

#### Conv1D CrossEntropy Implementation

**Date**: 2025-11-10  
**Status**: ✅ **COMPLETE** - Implementation + Comprehensive Tests

**Files Created**:
- ✅ `ai-analysis/conv1d/conv1d_cross_entropy_loss.py` - Conv1DClassifier + mapping utilities (163 lines)
- ✅ `ai-analysis/conv1d/trainer_cross_entropy_loss.py` - Conv1DCETrainer class (278 lines, with fallback imports)
- ✅ `ai-analysis/conv1d/conv1d_train_on_features_cross_entropy_loss.py` - Features CE script (350 lines)
- ✅ `ai-analysis/conv1d/conv1d_train_on_vectorized_essays_cross_entropy_loss.py` - Essays CE script (357 lines)
- ✅ `tests/test_conv1d_cross_entropy_loss_classifier.py` - 24 unit tests (architecture, forward, gradients, devices, masked pooling)
- ✅ `tests/test_conv1d_cross_entropy_loss_smoke.py` - 2 integration tests (1-epoch training, script imports)

**Design**: Mirrors BLSTM CE implementation exactly:
- Reuses BLSTM CE mapping utilities (scores_to_class_indices, logits_to_scores, validate_scores_for_ce)
- Conv1DClassifier with 6-logit classification head replacing regression head
- Same encoder structure as Conv1DRegressor (conv layers, batch norms, pooling)
- Handles both 2D features [B, F] and 3D sequences [B, L, D] with masked pooling

**Output Directories**:
- Features CE: `runs/features_cross_entropy_loss/conv1d_model/`
- Essays CE: `runs/vectorized_essays_cross_entropy_loss/conv1d_model/`

**Key Implementation Details**:
1. **Mapping Utilities**: Imported from BLSTM CE (NUM_CLASSES, scores_to_class_indices, etc.)
2. **Trainer**: Conv1DCETrainer mirrors BiLSTMCETrainer with Conv1D-specific details
3. **Input Handling**: Features pass lengths=None; sequences use masked pooling
4. **Metrics**: All computed in score space; best tracking by val MAE
5. **Configs**: Uses Conv1D ModelConfig/TrainConfig/SerializationConfig
6. **AMP**: torch.amp.GradScaler for compatibility with conv1d/trainer.py

**Test Results**:
- ✅ 26/26 tests passing (24 unit + 2 integration)
- ✅ Architecture, forward pass, gradients, device handling all verified
- ✅ Masked pooling behavior validated (with appropriate BatchNorm tolerances)
- ✅ 1-epoch smoke test completes successfully on 500-sample dataset
- ✅ Checkpoint metadata verified (loss_type: "CrossEntropyLoss")
- ✅ All predictions in valid C1 score set {0, 40, 80, 120, 160, 200}
- ✅ Script imports confirmed for both features and vectorized essays training

---

## Training Scripts Quick Reference

### BLSTM CrossEntropy Loss
```powershell
# Train on features
cd ai-analysis/blstm
python blstm_train_on_features_cross_entropy_loss.py

# Train on vectorized essays
python blstm_train_on_vectorized_essays_cross_entropy_loss.py
```

### Conv1D CrossEntropy Loss
```powershell
# Train on features
cd ai-analysis/conv1d
python conv1d_train_on_features_cross_entropy_loss.py

# Train on vectorized essays
python conv1d_train_on_vectorized_essays_cross_entropy_loss.py
```

**Output Directories**:
- BLSTM Features: `runs/features_cross_entropy_loss/blstm_model/`
- BLSTM Essays: `runs/vectorized_essays_cross_entropy_loss/blstm_model/`
- Conv1D Features: `runs/features_cross_entropy_loss/conv1d_model/`
- Conv1D Essays: `runs/vectorized_essays_cross_entropy_loss/conv1d_model/`

---

## Open Questions (Resolved)

1. ✅ **Option A vs B**: Resolved - using Option B (true classification head)
2. ✅ **Best checkpoint metric**: Confirmed - val MAE (same as regression)
3. ✅ **OOD labels**: Strict validation with validate_scores_for_ce() - raises on invalid
4. ✅ **Label smoothing**: Not used - default off
5. ✅ **Class weighting**: Not used - uniform weights
6. ✅ **Snap utility location**: Exists in common/metrics.py as `snap_to_step()`
7. ✅ **Scheduler/early stopping**: No coupling - loss name is metadata only
