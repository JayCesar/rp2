import concurrent.futures
import gc
import language_tool_python
import logging
import pathlib
import polars as pl
import spacy
import subprocess
import time
import spacy
import language_tool_python
from threading import Lock
import utils


logger = utils.logger

# Global progress tracking
progress_lock = Lock()
completed_essays = 0

# Configuration
TEST_MODE = True  # Set to False to process all essays
ROW_UPPER_LIMIT = (
    100 if TEST_MODE else None
)  # Process 25 essays in test mode, all otherwise
BATCH_SIZE = 200  # Process essays in larger batches (memory optimized)
MAX_WORKERS = 3  # Increase workers since we're using smaller models (~1.5GB total)

try:
    logger.info("Loading spaCy model 'pt_core_news_lg'...")
    nlp = spacy.load("pt_core_news_lg")
    logger.info("spaCy model loaded successfully")
except OSError:
    logger.info("spaCy model not found. Downloading 'pt_core_news_lg'...")
    subprocess.run(["python", "-m", "spacy", "download", "pt_core_news_lg"], check=True)
    nlp = spacy.load("pt_core_news_lg")
    logger.info("spaCy model downloaded and loaded successfully")

logger.info("Initializing LanguageTool for Portuguese (Brazil)...")
tool = language_tool_python.LanguageTool("pt-BR")
logger.info("LanguageTool initialized successfully")


def essay_metrics(essay_data, total_essay_count):
    essay_id = essay_data["essay_id"]
    if essay_id % 10 == 0:
        logger.info(f"Processing essay {essay_id}/{total_essay_count}...")

    essay = essay_data["essay_as_single_utf8_string"]

    # Basic text statistics
    doc = nlp(essay)
    word_count = len([token for token in doc if not token.is_punct])
    sentence_count = len([doc for doc in doc.sents])

    # LanguageTool error checking
    errors = tool.check(essay)

    # logger.info(f"Errors found in essay {essay_idx}: {errors}")

    error_counts = {}
    for error in errors:
        error_category = error.category
        if error_category in error_counts:
            error_counts[error_category] += 1
        else:
            error_counts[error_category] = 1

    total_error_count = sum(error_counts.values())

    return pl.DataFrame(
        error_counts
        | essay_data
        | {
            "total_error_count": total_error_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
        }
    )


def essay_token_count(encoded_essay):
    essay_tokens = encoded_essay["input_ids"][0]

    return len(essay_tokens)


def main():
    global completed_essays
    completed_essays = 0  # Reset progress counter

    logger.info("Starting LanguageTool feature extraction process...")

    dataset_parquet_file_path = (
        pathlib.Path.cwd()
        / "generated_datasets"
        / "extended_essay-br_preprocessed_for_LanguageTool.parquet"
    )
    logger.info(f"Checking dataset file: {dataset_parquet_file_path}")
    if not dataset_parquet_file_path.exists():
        logger.error(f"Dataset file not found at path {dataset_parquet_file_path}")
        return

    logger.info(f"Loading dataset from {dataset_parquet_file_path}...")
    relevant_columns = "c1", "essay_as_single_utf8_string", "prompt"
    dataset = (
        pl.scan_parquet(dataset_parquet_file_path)
        .select(relevant_columns)
        .drop_nulls()
        .unique()
        .with_row_index("essay_id")
    )

    # Apply row limit if specified
    if ROW_UPPER_LIMIT is not None:
        logger.info(f"Applying row limit: {ROW_UPPER_LIMIT}")
        dataset = dataset.head(ROW_UPPER_LIMIT)
        logger.info(f"Applied row limit. Processing at most {ROW_UPPER_LIMIT} essays")
    else:
        logger.info("No row limit applied. Processing all essays")
    dataset = dataset.collect()
    logger.info(f"Dataset loaded successfully. Shape: {dataset.shape}")

    total_essay_count = len(dataset)
    logger.info(f"Starting feature extraction for {total_essay_count} essays...")

    results = (
        essay_metrics(essay_data, total_essay_count)
        for essay_data in dataset.to_dicts()
    )

    logger.info("Concatenating results...")
    dataset_with_languagetool_metrics = pl.concat(
        results,
        how="diagonal",
    ).with_columns(pl.all().fill_null(strategy="zero"))

    logger.info(
        f"Feature extraction completed. Result shape: {dataset_with_languagetool_metrics.shape}"
    )
    logger.info(f"Final dataset preview:\n{dataset_with_languagetool_metrics.head()}")

    # Save results to files
    project_root = pathlib.Path(__file__).parent.parent.parent
    assert project_root.name == "rp2"

    generated_datasets_directory = project_root / "generated_datasets"
    generated_datasets_directory.mkdir(exist_ok=True)

    dataset_with_languagetool_metrics_filename = "dataset_with_languagetool_metrics"
    dataset_with_languagetool_metrics_extensions = "parquet", "csv", "json"

    utils.save_dataset(
        dataset_with_languagetool_metrics,
        dataset_with_languagetool_metrics_filename,
        *dataset_with_languagetool_metrics_extensions,
    )

    logger.info("LanguageTool feature extraction process completed successfully!")


if __name__ == "__main__":
    main()
