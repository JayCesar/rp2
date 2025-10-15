import pathlib
import re

import polars as pl
import utils

logger = utils.logger

spacy_model_name = "pt_core_news_lg"
nlp = utils.spacy_model(spacy_model_name)


def preprocess_text(text: str, idx: int) -> str:
    """Preprocess text using NLP techniques: lemmatization, stop word removal, cleaning.

    Args:
        text: Raw text to preprocess

    Returns:
        Preprocessed text string
    """
    if idx % 10 == 0:
        logger.info(f"preprocess_text: essay {idx}")

    text = utils.essay_line_to_single_utf8_string(text)

    # Remove URLs, emails, and special patterns
    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        text,
    )
    text = re.sub(r"\S+@\S+", "", text)

    # Process with spaCy
    doc = nlp(text)

    # Extract lemmatized tokens, removing stop words and non-alphabetic tokens
    processed_tokens = []
    for token in doc:
        # Skip stop words, punctuation, spaces, and very short tokens
        if (
            not token.is_stop
            and not token.is_punct
            and not token.is_space
            and not token.is_punct
            and len(token.text.strip()) > 1
        ):
            # Use lemma if available, otherwise use original token
            lemma = token.lemma_.lower().strip()
            if (
                lemma and lemma != "-PRON-"
            ):  # spaCy sometimes returns -PRON- for pronouns
                processed_tokens.append(lemma)
            else:
                processed_tokens.append(token.text.lower().strip())

    # Join tokens back into text
    processed_text = " ".join(processed_tokens)

    # If preprocessing resulted in empty text, return original
    if not processed_text.strip():
        return text

    return processed_text


def load_and_preprocess_dataset(
    csv_path: str | pathlib.Path, max_samples: int | None = None
) -> pl.DataFrame:
    """Load and preprocess the essay dataset.

    Args:
        csv_path: Path to the CSV file containing essays
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        Polars DataFrame with essays, essay prompts, and c1 scores
    """
    logger.info(f"Loading and preprocessing dataset from {csv_path}")

    DEFAULT_MAX_SAMPLE_SIZE = 2**31 - 1
    relevant_columns = ["c1", "essay", "prompt"]
    dataset = (
        pl.scan_csv(csv_path)
        .head(max_samples if max_samples is not None else DEFAULT_MAX_SAMPLE_SIZE)
        .select(relevant_columns)
        .drop_nulls()
        .unique()
        .with_columns(
            pl.col("essay")
            .map_batches(
                lambda essays: pl.Series(
                    (preprocess_text(essay, idx) for idx, essay in enumerate(essays))
                ),
                return_dtype=pl.Utf8,
            )
            .alias("essay_as_single_utf8_string")
        )
        .collect()
    )

    logger.info(
        f"Dataset loaded and preprocessed with {len(dataset)} samples:\n{dataset}"
    )

    return dataset


def main():
    project_root = pathlib.Path(__file__).parent.parent.parent
    assert project_root.name == "rp2"

    original_dataset_csv_path = project_root / "database" / "extended_essay-br.csv"

    dataset = load_and_preprocess_dataset(original_dataset_csv_path)
    logger.info(f"Dataset loaded with {len(dataset)} samples")

    generated_dataset_extensions = "csv", "parquet", "json"
    utils.save_dataset(
        dataset,
        "extended_essay-br_preprocessed_for_BERT",
        *generated_dataset_extensions,
    )


if __name__ == "__main__":
    main()
