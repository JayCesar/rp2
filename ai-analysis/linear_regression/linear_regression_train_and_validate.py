import numpy as np
import pathlib
import polars as pl
import sklearn
from typing import Optional


def load_and_prepare_data(csv_path: pathlib.Path, max_samples: Optional[int] = None):
    """Load and prepare the essay dataset.

    Args:
        csv_path: Path to the CSV file containing essays
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        numpy ndarray of the dataset in the provided path
    """
    print(f"Loading dataset from {csv_path}")
    print(f"[DEBUG] Loading dataset from {csv_path}")

    SAMPLE_SIZE_UPPER_BOUND = 2**31 - 1
    dataset = (
        pl.scan_csv(csv_path)
        .head(max_samples if max_samples is not None else SAMPLE_SIZE_UPPER_BOUND)
        .drop_nulls()
        .unique()
        .collect()
        .to_numpy()
    )

    return dataset


def main():
    project_root = pathlib.Path(__file__).parent.parent.parent
    csv_path = (
        project_root
        / "generated_datasets"
        / "dataset_with_languagetool_metrics_neuralmind--bert-base-portuguese-cased.csv"
    )

    dataset = load_and_prepare_data(csv_path)
    print(dataset)


if __name__ == "__main__":
    main()
