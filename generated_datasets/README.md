Large parquet split due to GitHub's 2 GiB per-file limit.

Parts (tracked via LFS):
- extended_essay-br_preprocessed_for_BLSTM_part1.parquet
- extended_essay-br_preprocessed_for_BLSTM_part2.parquet

Recreate original (skips if exists):
  uv run generated_datasets/fuse_blstm_parquet.py

Output:
- extended_essay-br_preprocessed_for_BLSTM.parquet (gitignored)

This fused BLSTM parquet is required by the vectorized-essay training scripts,
including the FocalLoss gamma-sweep entrypoints:

- Conv1D FocalLoss (vectorized essays):
  - `ai-analysis/conv1d/conv1d_train_on_vectorized_essays_focal_loss.py`
- BLSTM FocalLoss (vectorized essays):
  - `ai-analysis/blstm/blstm_train_on_vectorized_essays_focal_loss.py`

Both scripts will prefer the fused parquet when present. The BLSTM script also
attempts to run the fuse helper automatically when the fused file is missing,
but keeping this file up to date via the command above is recommended.
