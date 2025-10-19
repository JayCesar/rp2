import pathlib

import language_tool_python
import polars as pl
import utils
import feature_extraction

logger = utils.logger

# Configuration
MAX_SAMPLES = None  # Process 25 essays in test mode, all otherwise

spacy_model_name = "pt_core_news_md"
try:
    nlp = utils.spacy_model(spacy_model_name)
except OSError:
    logger.error(f"Failed to load spaCy model {spacy_model_name}")

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
    sentences = list(doc.sents)
    sentence_count = len(sentences)

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

    # Lexical diversity
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    features_spacy = {}
    if len(lemmas) > 0:
        features_spacy["LEXICAL_DIVERSITY"] = len(set(lemmas)) / len(lemmas)
    else:
        features_spacy["LEXICAL_DIVERSITY"] = 0

    # Sentence average length
    if sentence_count:
        sentence_lengths = [len(sentence) for sentence in sentences]
        features_spacy["AVERAGE_SENTENCE_LENGTH"] = sum(sentence_lengths) / len(
            sentence_lengths
        )
    else:
        features_spacy["AVERAGE_SENTENCE_LENGTH"] = 0

    COLLOQUIALISMS = ["mano", "tá ligado", "tipo assim", "né", "daora"]
    FORMAL_CONJUNCTIONS = [
        "ademais",
        "outrossim",
        "dessa forma",
        "portanto",
        "entretanto",
        "contudo",
    ]

    features_custom = {}
    features_custom["COLLOQUALISM_COUNT"] = sum(
        1 for exp in COLLOQUIALISMS if exp in essay
    )
    features_custom["FORMAL_CONJUNCTION_COUNT"] = sum(
        1 for con in FORMAL_CONJUNCTIONS if con in essay
    )

    return pl.DataFrame(
        error_counts
        | essay_data
        | {
            "TOTAL_ERROR_COUNT": total_error_count,
            "WORD_COUNT": word_count,
            "SENTENCE_COUNT": sentence_count,
        }
        | features_spacy
        | features_custom
    )


def essay_token_count(encoded_essay):
    essay_tokens = encoded_essay["input_ids"][0]

    return len(essay_tokens)


def main():
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
    if MAX_SAMPLES is not None:
        logger.info(f"Applying row limit: {MAX_SAMPLES}")
        dataset = dataset.head(MAX_SAMPLES)
        logger.info(f"Applied row limit. Processing at most {MAX_SAMPLES} essays")
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
