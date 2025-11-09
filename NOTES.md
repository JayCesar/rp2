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

---

## Open Questions (To Be Resolved)

1. ✅ **Option A vs B**: Resolved - using Option B (true classification head)
2. **Best checkpoint metric**: Confirm unchanged (likely MAE or QWK from regression system)
3. **OOD labels**: Handling for any labels outside {0,40,80,120,160,200} - assert/raise for now
4. **Label smoothing**: Default off; can be parameterized if needed later
5. **Class weighting**: Default uniform; can add if class imbalance is significant
6. **Snap utility location**: Exists in `blstm.py` as `snap_to_step()`
7. **Scheduler/early stopping coupling**: Check if loss name affects any logic (unlikely)
