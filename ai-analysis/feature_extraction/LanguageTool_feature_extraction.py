import concurrent.futures
import language_tool_python
import logging
import pathlib
import polars as pl
import spacy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROW_UPPER_LIMIT = None  # Set to None to use all samples


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

    dataset_parquet_file_path = (
        pathlib.Path.cwd()
        / "generated_datasets"
        / "extended_essay-br_preprocessed_for_LanguageTool.parquet"
    )
    if not dataset_parquet_file_path.exists():
        print(
            f"""[ERROR] Dataset file  not found at path {dataset_parquet_file_path}"""
        )
        return

    print(f"[DEBUG] Loading dataset from {dataset_parquet_file_path}...")
    relevant_columns = "c1", "essay_as_single_utf8_string", "prompt"
    dataset = (
        pl.scan_parquet(dataset_parquet_file_path)
        .select(relevant_columns)
        .drop_nulls()
        .unique()
        .collect()
    )
    print(f"[DEBUG] dataset loaded:\n{dataset}")

    def parallel_essay_metrics(essay_as_single_utf8_string_column, c1_column):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = (
                executor.submit(essay_metrics, essay, c1, idx, nlp, tool)
                for idx, (essay, c1) in enumerate(
                    zip(essay_as_single_utf8_string_column, c1_column)
                )
            )

            for future in concurrent.futures.as_completed(futures):
                yield future.result().collect()

    print(f"Calculating metrics for {len(dataset)} essays...")
    dataset_with_languagetool_metrics = pl.concat(
        parallel_essay_metrics(dataset["essay_as_single_utf8_string"], dataset["c1"]),
        how="diagonal",
    ).with_columns(pl.all().fill_null(strategy="zero"))
    print(
        "\n\n[DEBUG] dataset_with_languagetool_metrics:\n",
        dataset_with_languagetool_metrics,
    )

    def parallel_essay_metrics(dataset):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = (
                executor.submit(essay_metrics, essay, c1, idx, nlp, tool)
                for idx, (essay, c1) in enumerate(
                    zip(dataset["essay_as_single_utf8_string"], dataset["c1"])
                )
            )

            for future in concurrent.futures.as_completed(futures):
                yield future.result().collect()

    dataset_with_languagetool_metrics = pl.concat(
        parallel_essay_metrics(dataset),
        how="diagonal",
    ).with_columns(pl.all().fill_null(strategy="zero"))
    print(
        "\n\n[DEBUG] dataset_with_languagetool_metrics:\n",
        dataset_with_languagetool_metrics,
    )

    project_root = pathlib.Path(__file__).parent.parent.parent / "generated_datasets"

    dataset_with_languagetool_metrics_file_path_prefix = (
        "dataset_with_languagetool_metrics"
    )

    dataset_with_languagetool_metrics_parquet_file_path = (
        project_root / f"{dataset_with_languagetool_metrics_file_path_prefix}.parquet"
    )
    print(
        "[DEBUG] Writing dataset to Parquet file: ",
        dataset_with_languagetool_metrics_parquet_file_path,
    )
    dataset_with_languagetool_metrics.write_parquet(
        dataset_with_languagetool_metrics_parquet_file_path,
    )
    print(
        "[DEBUG] Metrics written to Parquet file: ",
        dataset_with_languagetool_metrics_parquet_file_path,
    )

    dataset_with_languagetool_metrics_csv_file_path = (
        project_root / f"{dataset_with_languagetool_metrics_file_path_prefix}.csv"
    )
    print(
        "[DEBUG] Writing dataset to CSV file: ",
        dataset_with_languagetool_metrics_csv_file_path,
    )
    dataset_with_languagetool_metrics.write_csv(
        dataset_with_languagetool_metrics_csv_file_path
    )
    print(
        "[DEBUG] Metrics written to CSV file: ",
        dataset_with_languagetool_metrics_csv_file_path,
    )

    dataset_with_languagetool_metrics_json_file_path = (
        project_root / f"{dataset_with_languagetool_metrics_file_path_prefix}.json"
    )
    print(
        "[DEBUG] Writing dataset to JSON file: ",
        dataset_with_languagetool_metrics_json_file_path,
    )
    dataset_with_languagetool_metrics.write_json(
        dataset_with_languagetool_metrics_json_file_path
    )
    print(
        "[DEBUG] Metrics written to JSON file: ",
        dataset_with_languagetool_metrics_json_file_path,
    )


if __name__ == "__main__":
    main()

