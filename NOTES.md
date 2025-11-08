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

### Existing BLSTM System
- `ai-analysis/blstm/blstm.py` - BiLSTMRegressor, configs, metrics, utilities
- `ai-analysis/blstm/blstm_training.py` - Training functions and main entry point
- `ai-analysis/blstm/trainer.py` - BiLSTMTrainer class (AMP, schedulers, early stopping)
- `ai-analysis/blstm/blstm_train_on_features.py` - Features training script
- `ai-analysis/blstm/blstm_train_on_vectorized_essays.py` - Vectorized essays training script
- `ai-analysis/common/` - Shared utilities (EssayDataset, collate_batch, etc.)

### New CE System (To Be Created)
- `ai-analysis/blstm/blstm_ce.py` - BiLSTMClassifier + mapping utilities
- `ai-analysis/blstm/blstm_training_ce.py` - CE training core
- `ai-analysis/blstm/trainer_ce.py` - BiLSTMCETrainer class
- `ai-analysis/blstm/blstm_train_on_features_ce.py` - Features CE script
- `ai-analysis/blstm/blstm_train_on_vectorized_essays_ce.py` - Vectorized essays CE script
- `tests/test_ce_mapping.py` - Mapping utilities tests
- `tests/test_bilstm_classifier.py` - BiLSTMClassifier tests
- `tests/test_training_ce.py` - Training core tests
- `tests/test_scripts_smoke.py` - Smoke tests for CE scripts

---

## Open Questions (To Be Resolved)

1. ✅ **Option A vs B**: Resolved - using Option B (true classification head)
2. **Best checkpoint metric**: Confirm unchanged (likely MAE or QWK from regression system)
3. **OOD labels**: Handling for any labels outside {0,40,80,120,160,200} - assert/raise for now
4. **Label smoothing**: Default off; can be parameterized if needed later
5. **Class weighting**: Default uniform; can add if class imbalance is significant
6. **Snap utility location**: Exists in `blstm.py` as `snap_to_step()`
7. **Scheduler/early stopping coupling**: Check if loss name affects any logic (unlikely)
