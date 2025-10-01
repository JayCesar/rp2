import logging
import pathlib

import numpy as np
import polars as pl
import scipy.stats
import sklearn.metrics
import sklearn.model_selection
import torch
import torch.nn as nn
import torch.utils.data
import tqdm
import transformers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration constants
MAX_LENGTH = 512  # Maximum sequence length for BERT tokenization

# C1 score normalization constants (from data analysis)
C1_MEAN = 135.45  # Without essays with C1 score of 0
# C1_MEAN = 133.25 # With essays with C1 score of 0

C1_STD = 28.63  # Without essays with C1 score of 0
# C1_STD = 33.16 # With essays with C1 score of 0

C1_MIN = 40  # Without essays with C1 score of 0
# C1_MIN = 0 # With essays with C1 score of 0

C1_MAX = 200

SAMPLE_SIZE_UPPER_BOUND = 2**31 - 1


def normalize_c1_scores(scores):
    """Normalize C1 scores to [0, 1] range for better training stability."""
    return sklearn.preproessing.StandardScaler().transform(scores)


def denormalize_c1_scores(scores):
    """Denormalize C1 scores to [0, 200] range for better training stability."""
    return sklearn.preproessing.StandardScaler().inverse_transform(scores)


def quadratic_weighted_kappa(y_true, y_pred, labels=None):
    """Calculate Quadratic Weighted Kappa (QWK) score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of possible labels (optional)

    Returns:
        QWK score between -1 and 1, where 1 is perfect agreement
    """
    if labels is None:
        labels = sorted(list(set(y_true + y_pred)))

    # Create confusion matrix
    n_labels = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    confusion_matrix = np.zeros((n_labels, n_labels))
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = label_to_idx[true_label]
        pred_idx = label_to_idx[pred_label]
        confusion_matrix[true_idx, pred_idx] += 1

    # Normalize to get observed agreement matrix
    total = confusion_matrix.sum()
    if total == 0:
        return 0.0

    observed_matrix = confusion_matrix / total

    # Calculate expected agreement matrix
    row_marginals = confusion_matrix.sum(axis=1) / total
    col_marginals = confusion_matrix.sum(axis=0) / total
    expected_matrix = np.outer(row_marginals, col_marginals)

    # Create quadratic weight matrix
    weights = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            weights[i, j] = (i - j) ** 2 / (n_labels - 1) ** 2

    # Calculate weighted agreements
    observed_agreement = np.sum(weights * observed_matrix)
    expected_agreement = np.sum(weights * expected_matrix)

    # Calculate QWK
    if expected_agreement == 0:
        return 0.0

    qwk = 1 - (observed_agreement / expected_agreement)
    return qwk


class BERTimbauForC1Prediction(nn.Module):
    """BERTimbau model with regression head for C1 prediction.

    Args:
        model_name: HuggingFace model name (e.g., 'neuralmind/bert-base-portuguese-cased')
        num_labels: Number of output labels (1 for regression)
        dropout_prob: Dropout probability for regularization
    """

    def __init__(self, model_name, num_labels: int = 1, dropout_prob: float = 0.1):
        super().__init__()  # Modern Python super() syntax
        self.bert = transformers.AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(self.bert.config.hidden_size, num_labels)

        # Initialize weights with Xavier/Glorot initialization for better training
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding tokens
            labels: Target C1 scores (optional, for training)

        Returns:
            Dictionary containing 'loss' and 'logits'
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use the [CLS] token representation (first token)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # Regression head
        logits = self.regressor(pooled_output)

        loss = None
        if labels is not None:
            # Use MSE loss for regression
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits.squeeze(-1), labels)

        return {"loss": loss, "logits": logits}


def load_and_prepare_data(
    csv_path: str | pathlib.Path,
    model,
    tokenizer,
    max_samples: int | None = None,
) -> pl.DataFrame:
    """Load and prepare the essay dataset.

    Args:
        csv_path: Path to the CSV file containing essays
        tokenizer: HuggingFace tokenizer
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        pl.DataFrame with vectorized_essays, essay prompts, and c1 scores
    """
    logger.info(f"Loading dataset from {csv_path}")
    print(f"[DEBUG] Loading dataset from {csv_path}")

    def vectorize_essay(essay, idx, total_essays, model, tokenizer):
        tokenized_essay = tokenizer(
            essay,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        if idx % 10 == 0:
            print(f"[DEBUG] vectorize_essay: essay {idx} / {total_essays} =\n{essay}")

        with torch.no_grad():
            # Get BERT embeddings using the underlying BERT model
            # instead of going through the custom model's forward method
            bert_outputs = model.bert(
                input_ids=tokenized_essay["input_ids"],
                attention_mask=tokenized_essay["attention_mask"],
            )
            # Use the [CLS] token representation (first token) from last hidden state
            all_token_vectors = bert_outputs.last_hidden_state
            cls_token_vector = all_token_vectors[0, 0]  # [batch_size=1, token_0=CLS]

        return cls_token_vector

    sample_max_amount = (
        max_samples if max_samples is not None else SAMPLE_SIZE_UPPER_BOUND
    )

    relevant_columns = ["c1", "essay_as_single_utf8_string", "prompt"]
    df = (
        pl.scan_csv(csv_path)
        .head(sample_max_amount)
        .select(relevant_columns)
        .drop_nulls()
        # .filter(
        #     (pl.col("c1") > 0)
        # ) # Remove samples with C1 score of 0, as they are not reliable enough
        .with_columns(
            pl.col("essay_as_single_utf8_string")
            .map_batches(
                lambda essays: pl.Series(
                    (
                        vectorize_essay(essay, idx, sample_max_amount, model, tokenizer)
                        for idx, essay in enumerate(essays)
                    )
                )
            )
            .alias("essay_vector")
        )
        .unique()
        .collect()
    )

    print(df)

    logger.info(f"Dataset loaded with {len(df)} samples")

    return df


def evaluate_model(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device
) -> dict[str, float]:
    """Evaluate the model and return metrics.

    Args:
        model: The trained model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on

    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            total_loss += outputs["loss"].item()

            # Get normalized predictions and denormalize them for metrics calculation
            normalized_preds = outputs["logits"].squeeze().cpu().numpy()
            denormalized_preds = denormalize_c1_scores(
                normalized_preds
                if hasattr(normalized_preds, "__len__")
                else [normalized_preds]
            )
            predictions.extend(denormalized_preds)

            # Denormalize true labels for metrics calculation
            normalized_labels = labels.cpu().numpy()
            denormalized_labels = denormalize_c1_scores(
                normalized_labels
                if hasattr(normalized_labels, "__len__")
                else [normalized_labels]
            )
            true_labels.extend(denormalized_labels)

    avg_loss = total_loss / len(dataloader)
    mse = sklearn.metrics.mean_squared_error(true_labels, predictions)
    mae = sklearn.metrics.mean_absolute_error(true_labels, predictions)
    r2 = sklearn.metrics.r2_score(true_labels, predictions)

    # Calculate Cohen's Kappa - round predictions and true labels to nearest C1 score levels
    # Convert continuous predictions to discrete C1 levels for kappa calculation
    def round_to_c1_levels(scores):
        """Round scores to nearest valid C1 levels (0, 40, 80, 120, 160, 200)"""
        c1_levels = [0, 40, 80, 120, 160, 200]
        rounded = []
        for score in scores:
            # Clamp to valid range first
            score = max(0, min(200, score))
            # Find closest C1 level
            closest_level = min(c1_levels, key=lambda x: abs(x - score))
            rounded.append(closest_level)
        return rounded

    true_labels_rounded = round_to_c1_levels(true_labels)
    predictions_rounded = round_to_c1_levels(predictions)

    try:
        kappa = sklearn.metrics.cohen_kappa_score(
            true_labels_rounded, predictions_rounded
        )
    except Exception:
        # If kappa calculation fails (e.g., only one class), set to 0
        kappa = 0.0

    # Calculate Quadratic Weighted Kappa
    try:
        qwk = quadratic_weighted_kappa(
            true_labels_rounded, predictions_rounded, labels=[0, 40, 80, 120, 160, 200]
        )
    except Exception:
        qwk = 0.0

    # Calculate Pearson correlation
    try:
        pearson_corr, pearson_p = scipy.stats.pearsonr(true_labels, predictions)
        # Handle case where correlation is NaN
        if np.isnan(pearson_corr):
            pearson_corr = 0.0
    except Exception:
        pearson_corr = 0.0

    return {
        "loss": avg_loss,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "kappa": kappa,
        "qwk": qwk,
        "pearson_corr": pearson_corr,
    }


def train_model(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    learning_rate: float = 2e-5,
) -> tuple[list[float], list[dict[str, float]]]:
    """Train the BERTimbau model for C1 prediction.

    Args:
        model: The model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        device: Device to train on
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer

    Returns:
        Tuple of (train_losses, validation_metrics)
    """
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps
    )

    logger.info(f"Starting training for {num_epochs} epochs")

    train_losses = []
    val_metrics = []

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        total_train_loss = 0
        train_progress = tqdm.tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")

        for batch in train_progress:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs["loss"]

            loss.backward()
            # Gradient clipping for stability (modern best practice)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            train_progress.set_postfix({"loss": loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation phase
        val_metrics_epoch = evaluate_model(model, val_dataloader, device)
        val_metrics.append(val_metrics_epoch)

        current_epoch = epoch + 1
        logger.info(f"Epoch {current_epoch} - Train Loss: {avg_train_loss:.4f}")
        logger.info(
            f"Epoch {current_epoch} - Val Loss: {val_metrics_epoch['loss']:.4f}"
        )
        logger.info(f"Epoch {current_epoch} - Val MSE: {val_metrics_epoch['mse']:.4f}")
        logger.info(f"Epoch {current_epoch} - Val MAE: {val_metrics_epoch['mae']:.4f}")
        logger.info(f"Epoch {current_epoch} - Val R²: {val_metrics_epoch['r2']:.4f}")
        logger.info(
            f"Epoch {current_epoch} - Val Kappa: {val_metrics_epoch['kappa']:.4f}"
        )
        logger.info(f"Epoch {current_epoch} - Val QWK: {val_metrics_epoch['qwk']:.4f}")
        logger.info(
            f"Epoch {current_epoch} - Val Pearson: {val_metrics_epoch['pearson_corr']:.4f}"
        )

    return train_losses, val_metrics


def main():
    """Main function to run the fine-tuning process."""
    # Quick device check
    print(f"\n{'=' * 50}")
    print("BERTimbau C1 Fine-tuning Starting")
    print(f"{'=' * 50}")

    # Configuration
    MODEL_NAME = "neuralmind/bert-base-portuguese-cased"  # BERTimbau
    MAX_LENGTH = 512  # BERT maximum sequence length
    BATCH_SIZE = 28  # Larger batch size for better GPU utilization
    NUM_EPOCHS = 10  # Fewer epochs may prevent overfitting
    LEARNING_RATE = 3.5e-5  # Scaled learning rate for batch size 28
    MAX_SAMPLES = 100  # Set to None to use all samples

    # Check and display device information first
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice Detection:")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU AVAILABLE - Will use: {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        import platform

        cpu_info = platform.processor() or "Unknown CPU"
        print(f"GPU NOT Available - Will use CPU: {cpu_info}")
        print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.device_count() > 0:
            print(f"Note: {torch.cuda.device_count()} GPU(s) detected but not usable")

    project_root = pathlib.Path(__file__).parent.parent.parent
    assert project_root.name == "rp2"

    csv_path = (
        project_root
        / "generated_datasets"
        / "extended_essay-br_preprocessed_for_BERT.csv"
    )
    model_save_path = project_root / "models" / "bertimbau_c1_finetuned"

    # Ensure model save directory exists
    model_save_path.mkdir(parents=True, exist_ok=True)

    # Device already configured above

    # Initialize model
    logger.info(f"Loading model from {MODEL_NAME}")
    print(f"[DEBUG] Loading model from {MODEL_NAME}")
    model = BERTimbauForC1Prediction(MODEL_NAME, num_labels=1)

    # Initialize tokenizer
    logger.info(f"Loading tokenizer from {MODEL_NAME}")
    print(f"[DEBUG] Loading tokenizer from {MODEL_NAME}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load and prepare data
    dataset = load_and_prepare_data(csv_path, model, tokenizer, max_samples=MAX_SAMPLES)
    print(dataset)

    essays = dataset["esssays"]
    c1_labels = dataset["c1_labels"]

    # Create stratification groups for highly imbalanced data
    # Group similar C1 scores to ensure balanced splits
    stratify_groups = [max(0, min(4, (label - 1) // 200)) for label in c1_labels]

    # Split the data with stratification
    train_essays, val_essays, train_labels, val_labels = (
        sklearn.model_selection.train_test_split(
            essays, c1_labels, test_size=0.3, random_state=42, stratify=stratify_groups
        )
    )

    total_samples = len(train_essays) + len(val_essays)
    logger.info(
        f"Training samples: {len(train_essays)} ({len(train_essays) / total_samples:.1%}), Validation samples: {len(val_essays)} ({len(val_essays) / total_samples:.1%}), Total samples: {total_samples}"
    )

    # For regression problems, class weights are not applicable
    # The model will learn to predict continuous C1 scores directly
    logger.info(
        "Using MSE loss for regression - class weights not applicable for continuous targets"
    )
    logger.info(
        f"Training configuration: BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE:.6f}"
    )

    train_dataset = pl.DataFrame(
        {"essay": train_essays, "c1": train_labels, "type": "train"}
    )
    val_dataset = pl.DataFrame({"essay": val_essays, "c1": val_labels, "type": "val"})

    essays_df = pl.concat(
        (
            train_dataset,
            val_dataset,
        )
    )

    essays_df.write_parquet(
        project_root
        / "generated_datasets"
        / "extended_essay-br_preprocessed_for_BERT.parquet"
    )
    logger.info(
        f"Dataset saved to {project_root / 'generated_datasets' / 'extended_essay-br_preprocessed_for_BERT.parquet'}"
    )
    print(
        f"Dataset saved to {project_root / 'generated_datasets' / 'extended_essay-br_preprocessed_for_BERT.parquet'}"
    )

    essays_df.write_csv(
        project_root
        / "generated_datasets"
        / "extended_essay-br_preprocessed_for_BERT.csv"
    )
    logger.info(
        f"Dataset saved to {project_root / 'generated_datasets' / 'extended_essay-br_preprocessed_for_BERT.csv'}"
    )
    print(
        f"Dataset saved to {project_root / 'generated_datasets' / 'extended_essay-br_preprocessed_for_BERT.csv'}"
    )

    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset.to_torch(), batch_size=BATCH_SIZE, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset.to_torch(), batch_size=BATCH_SIZE, shuffle=False
    )

    # Train the model
    train_losses, val_metrics = train_model(
        model,
        train_dataloader,
        val_dataloader,
        device,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
    )

    # Save the fine-tuned model with normalization parameters
    logger.info(f"Saving model to {model_save_path}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer": tokenizer,
            "model_name": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "train_losses": train_losses,
            "val_metrics": val_metrics,
            "normalization": {
                "c1_min": C1_MIN,
                "c1_max": C1_MAX,
                "c1_mean": C1_MEAN,
                "c1_std": C1_STD,
            },
        },
        model_save_path / "bertimbau_c1_model.pth",
    )

    # Save tokenizer separately
    tokenizer.save_pretrained(model_save_path / "tokenizer")

    # Final evaluation
    final_validation_metrics = evaluate_model(model, val_dataloader, device)
    final_validation_metrics_df = pl.DataFrame(final_validation_metrics)

    final_validation_metrics_df.write_csv(
        model_save_path / "final_validation_metrics.csv"
    )
    print(
        f"Final validation metrics saved to {model_save_path / 'final_validation_metrics.csv'}"
    )

    final_validation_metrics_df.write_parquet(
        model_save_path / "final_validation_metrics.parquet"
    )
    print(
        f"Final validation metrics saved to {model_save_path / 'final_validation_metrics.parquet'}"
    )

    print("Final validation metrics:")
    for metric, value in final_validation_metrics.items():
        print(f"{metric}: {value:.4f}")

    logger.info("Fine-tuning completed successfully!")

    print(f"\n{'=' * 50}")
    print("✅ Fine-tuning completed successfully!")
    print(f"{'=' * 50}")
    if torch.cuda.is_available():
        print(f"Trained using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Trained using CPU")
    print(f"Model saved to: {model_save_path}")


if __name__ == "__main__":
    main()
