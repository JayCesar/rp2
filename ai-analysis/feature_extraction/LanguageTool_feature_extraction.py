import concurrent.futures
import gc
import language_tool_python
import logging
import pathlib
import polars as pl
import spacy
import subprocess
import time
from threading import Lock

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global progress tracking
progress_lock = Lock()
completed_essays = 0

# Configuration
TEST_MODE = False  # Set to False to process all essays
ROW_UPPER_LIMIT = (
    25 if TEST_MODE else None
)  # Process 25 essays in test mode, all otherwise
BATCH_SIZE = 200  # Process essays in larger batches (memory optimized)
MAX_WORKERS = 3  # Increase workers since we're using smaller models (~1.5GB total)


def get_memory_usage():
    """Get current memory usage in GB using system commands."""
    try:
        # Try to get memory info from /proc/self/status
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # Extract memory in KB and convert to GB
                    memory_kb = int(line.split()[1])
                    return memory_kb / (1024**2)
        return 0.0  # Fallback
    except:
        return 0.0  # Fallback if unable to read


def essay_metrics_worker(essay_data):
    """Worker function for multiprocessing that initializes its own models."""
    essay, essay_c1_score, essay_idx, total_essays = essay_data

    # Initialize models inside worker process (avoids pickling issues)
    import spacy
    import language_tool_python
    import time

    start_time = time.time()

    try:
        # Load smaller spaCy model to reduce memory usage (lg->sm saves ~1.5GB per worker)
        import gc

        nlp = spacy.load("pt_core_news_sm")  # Much smaller model
        tool = language_tool_python.LanguageTool("pt-BR")

        # Basic text statistics
        doc = nlp(essay)
        word_count = len([token for token in doc if not token.is_punct])
        sentence_count = len(list(doc.sents))

        # LanguageTool error checking
        errors = tool.check(essay)
        error_counts = {}
        for error in errors:
            error_category = error.category
            if error_category in error_counts:
                error_counts[error_category] += 1
            else:
                error_counts[error_category] = 1

        total_error_count = sum(error_counts.values())

        # Clean up resources aggressively
        tool.close()
        del nlp, tool, doc, errors  # Explicit cleanup
        gc.collect()  # Force garbage collection

        processing_time = time.time() - start_time

        # Return simple dict instead of Polars LazyFrame (easier to pickle)
        result = error_counts.copy()
        result.update(
            {
                "c1": essay_c1_score,
                "total_error_count": total_error_count,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "processing_time_seconds": processing_time,
                "essay_index": essay_idx,
            }
        )

        # Log progress for significant milestones
        if (essay_idx + 1) % 50 == 0 or (essay_idx + 1) in [
            1,
            5,
            10,
            25,
            100,
            250,
            500,
        ]:
            logger.info(
                f"Completed essay {essay_idx + 1}/{total_essays} - "
                f"Processing time: {processing_time:.2f}s - Word count: {word_count} - Errors: {total_error_count}"
            )

        return result

    except Exception as e:
        logger.error(f"Error processing essay {essay_idx + 1}: {str(e)}")
        # Return minimal dict for failed essays
        return {
            "c1": essay_c1_score,
            "total_error_count": -1,  # -1 indicates processing error
            "word_count": 0,
            "sentence_count": 0,
            "processing_time_seconds": time.time() - start_time,
            "essay_index": essay_idx,
        }


def essay_token_count(encoded_essay):
    essay_tokens = encoded_essay["input_ids"][0]

    return len(essay_tokens)


def main():
    global completed_essays
    completed_essays = 0  # Reset progress counter

    logger.info("Starting LanguageTool feature extraction process...")

    try:
        logger.info("Loading spaCy model 'pt_core_news_sm' (memory optimized)...")
        nlp = spacy.load("pt_core_news_sm")
        logger.info("spaCy model loaded successfully")
    except OSError:
        logger.info("spaCy model not found. Downloading 'pt_core_news_sm'...")
        subprocess.run(
            ["python", "-m", "spacy", "download", "pt_core_news_sm"], check=True
        )
        nlp = spacy.load("pt_core_news_sm")
        logger.info("spaCy model downloaded and loaded successfully")

    logger.info("Initializing LanguageTool for Portuguese (Brazil)...")
    tool = language_tool_python.LanguageTool("pt-BR")
    logger.info("LanguageTool initialized successfully")

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
        .collect()
    )
    logger.info(f"Dataset loaded successfully. Shape: {dataset.shape}")

    # Apply row limit if specified
    if ROW_UPPER_LIMIT is not None:
        logger.info(f"Applying row limit: {ROW_UPPER_LIMIT}")
        dataset = dataset.head(ROW_UPPER_LIMIT)
        logger.info(f"Applied row limit. Processing {len(dataset)} essays")
    else:
        logger.info(f"No row limit applied. Processing all {len(dataset)} essays")

    def parallel_essay_metrics(dataset, batch_offset=0):
        batch_size = len(dataset)
        total_essays_in_full_dataset = 6577  # Known total from earlier
        logger.info(
            f"Starting parallel processing of {batch_size} essays (batch offset: {batch_offset})..."
        )
        start_time = time.time()

        # Prepare data for worker processes (avoid passing complex objects)
        essay_data_list = [
            (essay, c1, batch_offset + idx, total_essays_in_full_dataset)
            for idx, (essay, c1) in enumerate(
                zip(dataset["essay_as_single_utf8_string"], dataset["c1"])
            )
        ]

        # Use limited workers to stay under 12GB RAM (each worker ~2-3GB for models)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:
            # Submit all tasks to worker processes
            futures = {
                executor.submit(essay_metrics_worker, essay_data): batch_offset + idx
                for idx, essay_data in enumerate(essay_data_list)
            }

            logger.info(f"Submitted {len(futures)} tasks to executor")

            results = []
            completed_count = 0

            for future in concurrent.futures.as_completed(futures):
                try:
                    result_dict = future.result()
                    # Convert dict to DataFrame
                    df = pl.DataFrame([result_dict])
                    results.append(df)
                    completed_count += 1

                    # Log overall progress every 25 completions for batches
                    if completed_count % 25 == 0 or completed_count in [1, 10]:
                        elapsed_time = time.time() - start_time
                        avg_time_per_essay = elapsed_time / completed_count
                        remaining_essays = batch_size - completed_count
                        eta = remaining_essays * avg_time_per_essay

                        logger.info(
                            f"Batch progress: {completed_count}/{batch_size} essays completed "
                            f"({completed_count / batch_size * 100:.1f}%) - "
                            f"Elapsed: {elapsed_time:.1f}s - ETA: {eta:.1f}s"
                        )

                except Exception as e:
                    essay_idx = futures[future]
                    logger.error(f"Failed to process essay {essay_idx}: {str(e)}")
                    # Add a placeholder result for failed essays
                    error_df = pl.DataFrame(
                        [
                            {
                                "c1": 0,
                                "total_error_count": -1,
                                "word_count": 0,
                                "sentence_count": 0,
                                "processing_time_seconds": 0,
                                "essay_index": essay_idx,
                            }
                        ]
                    )
                    results.append(error_df)

            total_time = time.time() - start_time
            logger.info(
                f"Parallel processing completed in {total_time:.2f}s. "
                f"Average time per essay: {total_time / batch_size:.2f}s"
            )

            return results

    total_essays = len(dataset)
    logger.info(f"Starting feature extraction for {total_essays} essays...")

    # Process in batches to manage memory
    all_results = []
    for batch_start in range(0, total_essays, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_essays)
        batch_dataset = dataset[batch_start:batch_end]

        memory_before = get_memory_usage()
        logger.info(
            f"Processing batch {batch_start // BATCH_SIZE + 1}: essays {batch_start + 1}-{batch_end} (RAM: {memory_before:.1f}GB)"
        )

        batch_results = parallel_essay_metrics(batch_dataset, batch_start)
        all_results.extend(batch_results)

        # Force garbage collection after each batch
        gc.collect()
        memory_after = get_memory_usage()

        logger.info(
            f"Completed batch {batch_start // BATCH_SIZE + 1}, got {len(batch_results)} results (RAM: {memory_after:.1f}GB) - total results so far: {len(all_results)}"
        )

        # Memory usage warning
        if memory_after > 10.0:
            logger.warning(
                f"Memory usage high: {memory_after:.1f}GB - consider reducing batch size"
            )

    logger.info(f"All batches completed, got {len(all_results)} total results")

    logger.info("Concatenating results...")
    dataset_with_languagetool_metrics = pl.concat(
        all_results,
        how="diagonal",
    ).with_columns(pl.all().fill_null(strategy="zero"))

    logger.info(
        f"Feature extraction completed. Result shape: {dataset_with_languagetool_metrics.shape}"
    )
    logger.info(f"Final dataset preview:\n{dataset_with_languagetool_metrics.head()}")

    # Save results to files
    project_root = pathlib.Path(__file__).parent.parent.parent / "generated_datasets"
    project_root.mkdir(exist_ok=True)

    dataset_with_languagetool_metrics_file_path_prefix = (
        "dataset_with_languagetool_metrics"
    )

    # Save as Parquet (most efficient)
    dataset_with_languagetool_metrics_parquet_file_path = (
        project_root / f"{dataset_with_languagetool_metrics_file_path_prefix}.parquet"
    )
    logger.info(
        f"Writing dataset to Parquet file: {dataset_with_languagetool_metrics_parquet_file_path}"
    )
    dataset_with_languagetool_metrics.write_parquet(
        dataset_with_languagetool_metrics_parquet_file_path,
    )
    logger.info(
        f"Successfully wrote Parquet file: {dataset_with_languagetool_metrics_parquet_file_path}"
    )

    # Save as CSV (for readability)
    dataset_with_languagetool_metrics_csv_file_path = (
        project_root / f"{dataset_with_languagetool_metrics_file_path_prefix}.csv"
    )
    logger.info(
        f"Writing dataset to CSV file: {dataset_with_languagetool_metrics_csv_file_path}"
    )
    dataset_with_languagetool_metrics.write_csv(
        dataset_with_languagetool_metrics_csv_file_path
    )
    logger.info(
        f"Successfully wrote CSV file: {dataset_with_languagetool_metrics_csv_file_path}"
    )

    # Save as JSON (for web compatibility)
    dataset_with_languagetool_metrics_json_file_path = (
        project_root / f"{dataset_with_languagetool_metrics_file_path_prefix}.json"
    )
    logger.info(
        f"Writing dataset to JSON file: {dataset_with_languagetool_metrics_json_file_path}"
    )
    dataset_with_languagetool_metrics.write_json(
        dataset_with_languagetool_metrics_json_file_path
    )
    logger.info(
        f"Successfully wrote JSON file: {dataset_with_languagetool_metrics_json_file_path}"
    )

    logger.info("LanguageTool feature extraction process completed successfully!")


if __name__ == "__main__":
    main()
