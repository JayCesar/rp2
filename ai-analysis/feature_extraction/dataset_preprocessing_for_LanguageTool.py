import concurrent.futures
import logging
import os
import pathlib
import re

import language_tool_python
import polars as pl
import spacy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def essay_line_to_single_utf8_string(essay_line: str):
    """
    Sequence of transformations performed on each essay_line:
    1) Use eval() to turn the essay_line from a Python list string representation
       into a Python list
    2) Join all the sentences of the essay with a " " between them
    3) Remove all sequences of multiple (more than 1) whitespaces in a row and
       replace each of them with a single " "
    """

    return re.sub(r"\s\s+", " ", " ".join(eval(essay_line)))


def essay_metrics(essay, essay_c1_score, essay_idx, nlp, tool):
    # Basic text statistics
    doc = nlp(essay)
    word_count = len([token for token in doc if not token.is_punct])
    sentence_count = len(list(doc.sents))

    if essay_idx % 25 == 0:
        print(f"[DEBUG] essay_idx: {essay_idx}")

    errors = tool.check(essay)
    error_counts = {}
    for error in errors:
        error_category = error.category
        if error_category in error_counts:
            error_counts[error_category] += 1
        else:
            error_counts[error_category] = 1

    total_error_count = 0
    for error_count in error_counts.values():
        total_error_count += error_count

    return pl.LazyFrame(
        error_counts
        | {
            "c1": essay_c1_score,
            "total_error_count": total_error_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
        }
    )


def essay_token_count(encoded_essay):
    essay_tokens = encoded_essay["input_ids"][0]

    return len(essay_tokens)


def main():
    nlp = spacy.load("pt_core_news_lg")
    tool = language_tool_python.LanguageTool("pt-BR")

    dataset_file_path = pathlib.Path.cwd() / "database" / "extended_essay-br.csv"
    if not dataset_file_path.exists():
        print(f"""[ERROR] Dataset file  not found at path {dataset_file_path}""")

        return

    ideal_chunksize = (
        6577 // os.process_cpu_count() if os.process_cpu_count() is not None else 1
    )
    print(f"[DEBUG] ideal_chunksize for parallel processing: {ideal_chunksize}")

    def parallel_essay_line_to_single_utf8_string(essay_column):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            return pl.Series(
                executor.map(
                    essay_line_to_single_utf8_string,
                    essay_column,
                    chunksize=ideal_chunksize,
                )
            )

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
                parallel_essay_line_to_single_utf8_string,
                return_dtype=pl.Utf8,
            )
            .alias("essay_as_single_utf8_string")
        )
        .collect()
    )

    project_root = pathlib.Path(__file__).parent.parent.parent / "generated_datasets"

    print(
        f"[DEBUG] Writing dataset to {project_root / 'extended_essay-br_preprocessed_for_LanguageTool.parquet'}"
    )
    dataset.write_parquet(
        project_root / "extended_essay-br_preprocessed_for_LanguageTool.parquet"
    )
    print(
        f"[DEBUG] Dataset written to {project_root / 'extended_essay-br_preprocessed_for_LanguageTool.parquet'}"
    )

    print(
        f"[DEBUG] Writing dataset to {project_root / 'extended_essay-br_preprocessed_for_LanguageTool.csv'}"
    )
    dataset.write_csv(
        project_root / "extended_essay-br_preprocessed_for_LanguageTool.csv"
    )
    print(
        f"[DEBUG] Dataset written to {project_root / 'extended_essay-br_preprocessed_for_LanguageTool.csv'}"
    )

    print(
        f"[DEBUG] Writing dataset to {project_root / 'extended_essay-br_preprocessed_for_LanguageTool.json'}"
    )
    dataset.write_json(
        project_root / "extended_essay-br_preprocessed_for_LanguageTool.json"
    )
    print(
        f"[DEBUG] Dataset written to {project_root / 'extended_essay-br_preprocessed_for_LanguageTool.json'}"
    )


if __name__ == "__main__":
    main()
