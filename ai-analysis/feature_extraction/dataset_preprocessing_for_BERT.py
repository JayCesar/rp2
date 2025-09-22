import pathlib
import sys
from typing import Dict, List, Tuple, Optional, Union
import polars as pl
from scipy.stats import pearsonr
import logging
import re
import string
import spacy
from dataset_preprocessing_for_LanguageTool import essay_line_to_single_utf8_string

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy Portuguese model for NLP preprocessing
spacy_model_name = "pt_core_news_lg"
try:
    nlp = spacy.load(spacy_model_name)
except OSError:
    logger.error(f"Portuguese spaCy model not found. Please install with: python -m spacy download {spacy_model_name}")
    sys.exit(1)

def preprocess_text(text: str) -> str:
    """Preprocess text using NLP techniques: lemmatization, stop word removal, cleaning.

    Args:
        text: Raw text to preprocess

    Returns:
        Preprocessed text string
    """
    text = essay_line_to_single_utf8_string(text)

    # Remove URLs, emails, and special patterns
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    # Remove excessive punctuation (keep some for sentence structure)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)

    # Process with spaCy
    doc = nlp(text)

    # Extract lemmatized tokens, removing stop words and non-alphabetic tokens
    processed_tokens = []
    for token in doc:
        # Skip stop words, punctuation, spaces, and very short tokens
        if (not token.is_stop and
            not token.is_punct and
            not token.is_space and
            not token.is_punct and
            len(token.text.strip()) > 1):
            # Use lemma if available, otherwise use original token
            lemma = token.lemma_.lower().strip()
            if lemma and lemma != '-PRON-':  # spaCy sometimes returns -PRON- for pronouns
                processed_tokens.append(lemma)
            else:
                processed_tokens.append(token.text.lower().strip())

    # Join tokens back into text
    processed_text = ' '.join(processed_tokens)

    # If preprocessing resulted in empty text, return original
    if not processed_text.strip():
        return text

    return processed_text

def load_and_preprocess_dataset(csv_path: Union[str, pathlib.Path],
                          max_samples: Optional[int] = None) -> Tuple[List[str], List[float]]:
    """Load and preprocess the essay dataset.

    Args:
        csv_path: Path to the CSV file containing essays
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        Polars DataFrame with essays, essay prompts, and c1 scores
    """
    logger.info(f"Loading and preprocessing dataset from {csv_path}")
    print(f"[DEBUG] Loading and preprocessing dataset from {csv_path}")

    SAMPLE_SIZE_UPPER_BOUND = 2 ** 31 - 1
    relevant_columns = ["c1", "essay", "prompt"]
    df = (
        pl.scan_csv(csv_path)
        .head(max_samples if max_samples is not None else SAMPLE_SIZE_UPPER_BOUND)
        .select(relevant_columns)
        .drop_nulls()
        # .filter((pl.col("c1") > 0)) # Remove samples with C1 score of 0, as they are not reliable enough
        .unique()
        .with_columns(
            pl.col("essay")
            .map_batches(
                lambda essay_column: pl.Series(
                    (
                        preprocess_text(essay_line)
                        for essay_line in essay_column
                    )
                ),
                return_dtype=pl.Utf8,
            )
            .alias("essay_as_single_utf8_string")
        )
        .collect()
    )

    logger.info(f"Dataset loaded and preprocessed with {len(df)} samples")
    print(f"[DEBUG] Dataset loaded and preprocessed with {len(df)} samples")

    return df

def main():
    project_root = pathlib.Path(__file__).parent.parent.parent
    original_dataset_csv_path = project_root / "database" / "extended_essay-br.csv"

    dataset = load_and_preprocess_dataset(original_dataset_csv_path)
    logger.info(f"Dataset loaded with {len(dataset)} samples")
    print(f"[DEBUG] Dataset loaded with {len(dataset)} samples")

    generated_datasets_path = project_root / "generated_datasets"
    generated_datasets_file_name = "extended_essay-br_preprocessed_for_BERT"

    logger.info(f"Writing preprocessed dataset to {generated_datasets_path}")
    print(f"[DEBUG] Writing preprocessed dataset to {generated_datasets_path}")
    preprocessed_dataset_csv_path = generated_datasets_path / f"{generated_datasets_file_name}.csv"
    dataset.write_csv(preprocessed_dataset_csv_path)
    logger.info(f"Dataset written to CSV file: {preprocessed_dataset_csv_path}")
    print(f"[DEBUG] Dataset written to CSV file: {preprocessed_dataset_csv_path}")

    logger.info(f"Writing preprocessed dataset to {generated_datasets_path}")
    print(f"[DEBUG] Writing preprocessed dataset to {generated_datasets_path}")
    preprocessed_dataset_json_path = generated_datasets_path / f"{generated_datasets_file_name}.json"
    dataset.write_json(preprocessed_dataset_json_path)
    logger.info(f"Dataset written to JSON file: {preprocessed_dataset_json_path}")
    print(f"[DEBUG] Dataset written to JSON file: {preprocessed_dataset_json_path}")

    logger.info(f"Writing preprocessed dataset to {generated_datasets_path}")
    print(f"[DEBUG] Writing preprocessed dataset to {generated_datasets_path}")
    preprocessed_dataset_parquet_path = generated_datasets_path / f"{generated_datasets_file_name}.parquet"
    dataset.write_parquet(preprocessed_dataset_parquet_path)
    logger.info(f"Dataset written to Parquet file: {preprocessed_dataset_parquet_path}")
    print(f"[DEBUG] Dataset written to Parquet file: {preprocessed_dataset_parquet_path}")

if __name__ == "__main__":
    main()
