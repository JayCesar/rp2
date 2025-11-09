1.Consult PyTorch docs via Context7 MCP; decide A vs B
- Query Context7 MCP for: "PyTorch CrossEntropyLoss best practices", "classification vs regression heads", "snapping continuous outputs for CE".
- Decision rule: proceed with Option B (true classification head with 6 logits, CE on class indices) unless docs explicitly endorse training a regressor and snapping for CE (unlikely).
- Record brief finding in repo (e.g., NOTES.md) and proceed with B now.
2. Audit current blstm code to mirror structure
- Read ai-analysis/blstm/blstm_training.py, blstm_train_on_features.py, blstm_train_on_vectorized_essays.py.
- Identify: BiLSTMRegressor (encoder + head), ModelConfig/TrainConfig/SerializationConfig, dataset/featurizers, snap mechanism, metric functions (MAE, RMSE, Kappa, QWK, step_accuracy, Pearson), run directory layout, checkpointing, logging.
- Note how targets are provided in batches (tensor dtype/shape) and where snapping is applied today.
3. Add class mapping utilities (score ↔ class index)
- Implement in a small shared module or inside blstm_training_ce.py:
  - score_to_class_idx(score): 0→0, 40→1, 80→2, 120→3, 160→4, 200→5.
  - class_idx_to_score(idx): 0→0, 1→40, 2→80, 3→120, 4→160, 5→200.
  - Vectorized variants for torch.Tensor (CPU/GPU), with type hints; validate values and raise/assert when out-of-domain; optionally snap-to-closest bucket for noisy inputs if current pipeline does that upstream.
4. Implement BiLSTMClassifier that reuses the BLSTM encoder
- Goal: Mirror BiLSTMRegressor but replace final regression head with a classification head producing 6 logits.
- Preferred reuse path:
  - If BiLSTMRegressor exposes its encoder (e.g., lstm + pooling) as a submodule or separable forward features path, subclass/wrap it to reuse the encoder and attach a new torch.nn.Linear(hidden_dim, 6).
  - Else, copy minimal encoder construction from BiLSTMRegressor to new class to avoid modifying existing code.
- Forward returns raw logits with shape [B, 6].
- Add helper: logits_to_scores(logits): argmax → class idx → score via mapping.
- Keep dropout/init to match regressor; keep typing hints and device-agnostic behavior.
5. Create core CE trainer: ai-analysis/blstm/blstm_training_cross_entropy_loss.py
- Structure mirrors blstm_training.py but for classification:
  - Build model from ModelConfig; use BiLSTMClassifier.
  - Criterion: nn.CrossEntropyLoss() on raw logits and LongTensor target indices.
  - train_one_epoch_ce:
    - Batch: X, y_scores → y_cls = score_to_class_idx(y_scores).long().
    - Forward: logits = model(X); loss = CE(logits, y_cls).
    - Backprop, step, clip, log.
  - evaluate_ce:
    - Forward to logits; y_pred_scores = logits_to_scores(logits).
    - Compute existing metrics by comparing y_pred_scores vs ground-truth y_scores:
      - snap if existing eval expects snapped scores (normally already bucketed).
      - Reuse MAE, RMSE, Kappa, QWK, step_accuracy, Pearson as-is.
  - fit_ce:
    - Loop epochs; track best metric (reuse same selection criterion as regression, or MAE/QWK per current pipeline).
    - Save checkpoints, logs, and metrics JSON; respect SerializationConfig.
  - Preserve seeds, deterministic flags, AMP/mixed precision (if used), gradient clipping, LR sched, early stopping behaviors from original trainer.
- Type hints everywhere; minimal duplication; shared utilities imported from existing modules where possible.
6. Create features CE entry script: blstm_train_on_features_cross_entropy_loss.py
- Mirror blstm_train_on_features.py:
  - Parse same CLI/Config; add brief description noting it uses CrossEntropyLoss.
  - Build feature datasets/dataloaders identically.
  - Call CE training entrypoints from blstm_training_ce.py.
  - Default run/output dir: runs/features_ce/.
  - Preserve experiment naming, tensorboard/logging hooks.
7. Create vectorized essays CE entry script: blstm_train_on_vectorized_essays_cross_entropy_loss.py
- Mirror blstm_train_on_vectorized_essays.py:
  - Same configs and loaders for embeddings.
  - Use CE trainer functions.
  - Default run/output dir: runs/vectorized_essays_ce/.
8. Ensure compatibility with existing configs and serialization
- Accept/forward ModelConfig, TrainConfig, SerializationConfig without breaking changes.
- If TrainConfig currently encodes loss type, set/override to "ce" in these scripts; otherwise just log "CrossEntropyLoss" in run metadata.
- Save model state_dict, optimizer/scheduler states, config JSON in new run dirs; do not overwrite regression runs.
9. Integrate class conversions in training/eval carefully
- Training:
  - Convert scores → class indices only for loss computation; keep original scores for reporting.
  - Targets dtype must be torch.long with shape [B].
- Evaluation:
  - Convert logits → argmax class idx → scores.
  - Compute all existing metrics on score space; keep any snapping logic consistent with current evaluation (likely no-op for classification outputs).
10. Preserve and label all outputs; distinguish from regression
- Run dirs:
  - features_ce/ and vectorized_essays_ce/ under runs/, mirroring the regression layout.
- File names/metadata:
  - Include "ce" in run name/metrics file; log "loss: CrossEntropyLoss" prominently.
- Keep parity in: metrics.json, predictions.csv, checkpoints, tensorboard summaries.
11. Quality checks and smoke tests
- Unit tests (lightweight or ad-hoc assertions):
  - Mapping functions roundtrip: score → idx → score.
  - Loss shape/dtype correctness: logits [B,6], targets [B] long.
- Dry run on a tiny subset:
  - One epoch; confirm non-NaN loss, metrics computed, files saved in new dirs.
  - Validate predictions.csv contains integer scores from {0,40,80,120,160,200}.
12. Documentation and code comments
- Add top-of-file docstrings in new scripts stating they use CrossEntropyLoss for classification.
- Brief note in README/ai-analysis/blstm section about new CE scripts, run dirs, and the class mapping.
13. Option A fallback (only if PyTorch docs say so)
- If MCP/PyTorch recommends against classification head and for regressor snapping (unlikely), implement BiLSTMSnappedRegressorClassifier:
  - Delegate to BiLSTMRegressor forward to get continuous score; snap to nearest bucket; map to class index for loss.
  - Warning: CrossEntropyLoss expects logits; snapped outputs are not logits, so would need a surrogate (e.g., soft labeling or distance-based logits), which is nonstandard; prefer B unless explicitly advised.
- Given constraints, keep this as a documented alternative; do not implement unless required.
14. Deliverables and commits
- New files:
  - ai-analysis/blstm/blstm_training_cross_entropy_loss.py
  - ai-analysis/blstm/blstm_train_on_features_cross_entropy_loss.py
  - ai-analysis/blstm/blstm_train_on_vectorized_essays_cross_entropy_loss.py
- Minimal diffs to other modules only if needed for reuse (no breaking changes).
- Commit messages (concise):
  - feat(blstm): add CE classifier + training core
  - feat(blstm): CE training on features (runs/features_ce)
  - feat(blstm): CE training on vectorized essays (runs/vectorized_essays_ce)
15. Open questions
- Confirm Option B is acceptable if MCP access fails?
- Which metric governs best checkpoint now (unchanged)?
- Any labels outside {0,40,80,120,160,200} to handle?
- Keep label smoothing off or add (default off)?
- Any class weighting for imbalance (needed)?
- Exact location/name of snap utility to reuse?
- Any custom scheduler/early stopping coupling to loss name? 
