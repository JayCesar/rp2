import pathlib
import subprocess
import sys

import polars as pl
import torch
import transformers
import utils

logger = utils.logger

# Configuration
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"  # BERTimbau
MAX_LENGTH = 512  # BERT maximum sequence length
MAX_SAMPLES = None  # Set to None to use all samples

# Initialize model
logger.info(f"Loading model from {MODEL_NAME}")
model = transformers.AutoModel.from_pretrained(MODEL_NAME)
logger.info("model loaded")

# Initialize tokenizer
logger.info(f"Loading tokenizer from {MODEL_NAME}")
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
logger.info("tokenizer loaded")

project_root = pathlib.Path(__file__).parent.parent.parent
assert project_root.name == "rp2"


def essay_token_count(encoded_essay):
    essay_tokens = encoded_essay["input_ids"][0]

    return len(essay_tokens)


def vectorize_essay(essay, idx):
    if idx % 10 == 0:
        logger.info(f"vectorize_essay: essay {idx}")

    tokenized_essay = tokenizer.encode(
        essay,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        # Get BERT embeddings using the underlying BERT model
        # instead of going through the custom model's forward method
        bert_outputs = model(tokenized_essay)
        # # Use the [CLS] token representation (first token) from last hidden state
        cls_token_vector = bert_outputs.pooler_output

    return cls_token_vector


def load_and_preprocess_dataset(csv_path) -> pl.DataFrame:
    logger.info(f"Loading dataset from {csv_path}...")

    relevant_columns = ["c1", "essay_as_single_utf8_string", "prompt"]
    DEFAULT_MAX_SAMPLE_SIZE = 2**31 - 1
    dataset = (
        pl.scan_csv(csv_path)
        .select(relevant_columns)
        .head(MAX_SAMPLES if MAX_SAMPLES is not None else DEFAULT_MAX_SAMPLE_SIZE)
        .drop_nulls()
        .unique()
    )
    logger.info(f"Dataset loaded from {csv_path}\n")

    logger.info("Vectorizing dataset's essays...")
    dataset = dataset.with_columns(
        pl.col("essay_as_single_utf8_string")
        .map_batches(
            lambda essays: pl.Series(
                (vectorize_essay(essay, idx) for idx, essay in enumerate(essays))
            ),
            return_dtype=pl.Array(pl.Float32, 768),
        )
        .alias("essay_vector")
    ).collect()
    logger.info(f"Dataset with {len(dataset)} samples:\n{dataset.head(10)}")

    return dataset


def main():
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

    dataset_with_vectorized_essays = load_and_preprocess_dataset(csv_dataset_file_path)

    generated_dataset_extensions = "parquet", "json"
    utils.save_dataset(
        dataset_with_vectorized_essays,
        "extended_essay-br_preprocessed_for_BLSTM",
        *generated_dataset_extensions,
    )


if __name__ == "__main__":
    main()
