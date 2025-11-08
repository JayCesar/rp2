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
MAX_LENGTH = 128  # Use 128 tokens to match BLSTM preprocessed dataset
MAX_SAMPLES = None  # Set to None to use all samples

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Initialize model
logger.info(f"Loading model from {MODEL_NAME}")
model = transformers.AutoModel.from_pretrained(MODEL_NAME)
model = model.to(device)
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

    # Use tokenizer() to get both input_ids and attention_mask
    tokenized_inputs = tokenizer(
        essay,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",  # Pad to MAX_LENGTH for consistent sequence length
        return_tensors="pt",
    )

    # Move inputs to device
    tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}

    with torch.no_grad():
        # Get BERT embeddings for all tokens
        bert_outputs = model(**tokenized_inputs)
        # Use all token embeddings from last hidden state
        # Shape: [1, MAX_LENGTH, 768] = [1, 128, 768]
        all_token_embeddings = bert_outputs.last_hidden_state

        # Remove batch dimension and return as numpy array: [128, 768]
        token_embeddings = all_token_embeddings.squeeze(0).cpu().numpy()

    return token_embeddings


def load_and_preprocess_dataset(csv_path) -> pl.DataFrame:
    logger.info(f"Loading dataset from {csv_path}...")
    if MAX_SAMPLES is not None:
        logger.info(f"Applying row limit: {MAX_SAMPLES}")
    else:
        logger.info("No row limit applied. Processing all essays")

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
    # Build both columns lazily and collect once
    dataset = (
        dataset.with_columns(
            pl.col("essay_as_single_utf8_string")
            .map_batches(
                lambda essays: pl.Series(
                    values=(
                        vectorize_essay(essay, idx).tolist()
                        for idx, essay in enumerate(essays)
                    ),
                    dtype=pl.Array(pl.Array(pl.Float32, 768), MAX_LENGTH),
                ),
                return_dtype=pl.Array(
                    pl.Array(pl.Float32, 768), MAX_LENGTH
                ),  # [128, 768]
            )
            .alias("essay_token_embeddings")  # token-level embeddings
        )
        .with_columns(
            pl.col("essay_as_single_utf8_string")
            .map_batches(
                lambda essays: pl.Series(
                    values=(
                        int(
                            tokenizer(
                                essay,
                                truncation=True,
                                max_length=MAX_LENGTH,
                                padding="max_length",
                                return_tensors="pt",
                            )["attention_mask"]
                            .sum()
                            .item()
                        )
                        for idx, essay in enumerate(essays)
                    ),
                    dtype=pl.Int32,
                ),
                return_dtype=pl.Int32,
            )
            .alias("essay_token_length")
        )
        .collect()
    )
    logger.info(f"Dataset with {len(dataset)} samples:\n{dataset}")

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

    generated_dataset_extension = "parquet"
    utils.save_dataset(
        dataset_with_vectorized_essays,
        "extended_essay-br_preprocessed_for_BLSTM",
        generated_dataset_extension,
    )


if __name__ == "__main__":
    main()
