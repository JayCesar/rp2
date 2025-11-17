#!/usr/bin/env python3
"""
Build histogram tables from prediction CSVs with score, target, and pred_snapped columns.

Inputs default to:
- ai-analysis\blstm\runs\vectorized_essays\blstm_model\validation_predictions_best.csv
- ai-analysis\blstm\runs\features\blstm_model\validation_predictions_best.csv

For each input file, this script writes one CSV in the same directory:
- <stem>_histogram_table.csv with columns: score, target, pred_snapped
  where score contains all possible values, target contains counts of original target values,
  and pred_snapped contains counts of original pred_snapped values

Usage examples:
  python make_histogram_tables.py
  python make_histogram_tables.py --inputs path\to\a.csv path\to\b.csv
"""
from pathlib import Path
import argparse
import sys

import pandas as pd

ALLOWED_SCORES = [0, 40, 80, 120, 160, 200]


def process_file(input_path: Path) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    required_cols = {"target", "pred_snapped"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in {input_path}: {sorted(missing)}")

    # Coerce to numeric, drop rows with non-numeric target
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df["pred_snapped"] = pd.to_numeric(df["pred_snapped"], errors="coerce")

    # Keep only rows where target is one of the allowed scores
    df = df[df["target"].isin(ALLOWED_SCORES)].copy()

    # Get target counts for each score
    target_counts = (
        df["target"].value_counts(dropna=False)
        .reindex(ALLOWED_SCORES, fill_value=0)
    )
    
    # Get pred_snapped counts for each score
    pred_counts = (
        df["pred_snapped"].value_counts(dropna=False)
        .reindex(ALLOWED_SCORES, fill_value=0)
    )
    
    # Create the combined dataframe with the requested layout
    combined_df = pd.DataFrame({
        "score": ALLOWED_SCORES,
        "target": [target_counts[score] for score in ALLOWED_SCORES],
        "pred_snapped": [pred_counts[score] for score in ALLOWED_SCORES]
    })

    out_dir = input_path.parent
    stem = input_path.stem
    output_path = out_dir / f"{stem}_histogram_table.csv"

    combined_df.to_csv(output_path, index=False)

    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build frequency tables for histogramming.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            r"ai-analysis\blstm\runs\vectorized_essays\blstm_model\validation_predictions_best.csv",
            r"ai-analysis\blstm\runs\features\blstm_model\validation_predictions_best.csv",
        ],
        help="Input CSV file paths containing 'target' and 'pred_snapped' columns.",
    )

    args = parser.parse_args(argv)

    exit_code = 0
    for inp in args.inputs:
        p = Path(inp)
        if not p.is_absolute():
            p = Path.cwd() / p
        try:
            output_path = process_file(p)
            print(f"Wrote: {output_path}")
        except Exception as e:
            print(f"Error processing {p}: {e}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
