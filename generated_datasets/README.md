Large parquet split due to GitHub's 2 GiB per-file limit.

Parts (tracked via LFS):
- extended_essay-br_preprocessed_for_BLSTM_part1.parquet
- extended_essay-br_preprocessed_for_BLSTM_part2.parquet

Recreate original (skips if exists):
  uv run generated_datasets/fuse_blstm_parquet.py

Output:
- extended_essay-br_preprocessed_for_BLSTM.parquet (gitignored)
