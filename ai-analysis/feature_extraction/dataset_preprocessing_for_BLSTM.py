import pathlib
import subprocess
import sys

import numpy as np
import polars as pl
import transformers
import utils

logger = utils.logger

ROW_UPPER_LIMIT = None  # Set to None to use all samples

# Configuration
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"  # BERTimbau
MAX_LENGTH = 512  # BERT maximum sequence length
BATCH_SIZE = 28  # Larger batch size for better GPU utilization
NUM_EPOCHS = 10  # Fewer epochs may prevent overfitting
LEARNING_RATE = 3.5e-5  # Scaled learning rate for batch size 28
MAX_SAMPLES = None  # Set to None to use all samples

project_root = pathlib.Path(__file__).parent.parent.parent
assert project_root.name == "rp2"


def essay_token_count(encoded_essay):
    essay_tokens = encoded_essay["input_ids"][0]

    return len(essay_tokens)


def vectorize_essay(essay, idx, model, tokenizer):
    if idx % 10 == 0:
        logger.info(f"vectorize_essay: essay {idx}")

    tokenized_essay = tokenizer.encode(
        essay,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # The vector representation of the [CLS] token is used to represent the
    # entire essay
    all_token_vectors = model(tokenized_essay)[0]
    cls_token_vector = (
        all_token_vectors[0][0].detach().numpy()
    )  # [0] for batch, then [0] for CLS token

    return cls_token_vector


def load_and_preprocess_dataset(csv_path, model, tokenizer) -> pl.DataFrame:
    logger.info(f"Loading dataset from {csv_path}...")

    relevant_columns = ["c1", "essay_as_single_utf8_string", "prompt"]
    DEFAULT_MAX_SAMPLE_SIZE = 2**31 - 1
    dataset = (
        pl.scan_csv(csv_path)
        .select(relevant_columns)
        .head(MAX_SAMPLES if MAX_SAMPLES is not None else DEFAULT_MAX_SAMPLE_SIZE)
        .drop_nulls()
        .unique()
        .with_columns(
            pl.col("essay_as_single_utf8_string")
            .map_batches(
                lambda essays: pl.Series(
                    (
                        vectorize_essay(essay, idx, model, tokenizer)
                        for idx, essay in enumerate(essays)
                    )
                ),
                return_dtype=pl.Array(pl.Float32, 768),
            )
            .alias("essay_vector")
        )
        .collect()
    )
    logger.info(f"Dataset loaded ({len(dataset)} lines):\n{dataset}")

    return dataset


def main():
    # Initialize model
    logger.info(f"Loading model from {MODEL_NAME}")
    model = transformers.AutoModel.from_pretrained(MODEL_NAME, num_labels=1)
    logger.info("model loaded")

    # Initialize tokenizer
    logger.info(f"Loading tokenizer from {MODEL_NAME}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.info("tokenizer loaded")

    csv_dataset_file_path = (
        project_root
        / "generated_datasets"
        / "extended_essay-br_preprocessed_for_BERT.csv"
    )
    if not csv_dataset_file_path.exists():
        logger.error(f"Dataset file  not found at path {csv_dataset_file_path}")
        logger.info(f"Generating necessary {csv_dataset_file_path} dataset...")
        try:
            subprocess.run(
                [sys.executable, "dataset_preprocessing_for_BERT.py"],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(
                f"Successfully generated and saved dataset at path: {csv_dataset_file_path}"
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate dataset at {csv_dataset_file_path}: {e}")
            logger.error(f"Command output: {e.stdout}")
            logger.error(f"Command error: {e.stderr}")
            raise

    dataset_with_vectorized_essays = load_and_preprocess_dataset(
        csv_dataset_file_path, model, tokenizer
    )

    generated_dataset_extensions = "parquet", "json"
    utils.save_dataset(
        dataset_with_vectorized_essays,
        "extended_essay-br_preprocessed_for_BLSTM",
        *generated_dataset_extensions,
    )


if __name__ == "__main__":
    main()
