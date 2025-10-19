import pathlib
import sys

import language_tool_python
import polars as pl
import spacy
import utils

logger = utils.logger


def essay_token_count(encoded_essay):
    essay_tokens = encoded_essay["input_ids"][0]

    return len(essay_tokens)


def main():
    dataset_file_path = pathlib.Path.cwd() / "database" / "extended_essay-br.csv"
    if not dataset_file_path.exists():
        logger.error(f"Dataset file not found at path {dataset_file_path}")

        return

    relevant_columns = "c1", "essay", "prompt"
    max_rows = None  # Set to None to use all samples
    DEFAULT_MAX_SAMPLE_SIZE = 2**31 - 1
    dataset = (
        pl.scan_csv(dataset_file_path)
        .head(max_rows if max_rows is not None else DEFAULT_MAX_SAMPLE_SIZE)
        .select(relevant_columns)
        .drop_nulls()
        .unique()
        .with_columns(
            pl.col("essay")
            .map_batches(
                lambda essays: pl.Series(
                    (utils.essay_line_to_single_utf8_string(essay) for essay in essays)
                ),
                return_dtype=pl.Utf8,
            )
            .alias("essay_as_single_utf8_string")
        )
        .collect()
    )

    extensions = "parquet", "csv", "json"
    utils.save_dataset(
        dataset, "extended_essay-br_preprocessed_for_LanguageTool", *extensions
    )


if __name__ == "__main__":
    main()
