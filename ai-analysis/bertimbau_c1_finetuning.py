import os
import pathlib
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import pearsonr
import numpy as np
import transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer,
)
import logging
from tqdm import tqdm
import re
import string
import spacy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration constants
MAX_LENGTH = 512  # Maximum sequence length for BERT tokenization

# C1 score normalization constants (from data analysis)
C1_MEAN = 135.45 # Without essays with C1 score of 0
# C1_MEAN = 133.25 # With essays with C1 score of 0

C1_STD = 28.63 # Without essays with C1 score of 0
# C1_STD = 33.16 # With essays with C1 score of 0

C1_MIN = 40 # Without essays with C1 score of 0
# C1_MIN = 0 # With essays with C1 score of 0

C1_MAX = 200


def normalize_c1_scores(scores):
    """Normalize C1 scores to [0, 1] range for better training stability."""
    return [(score - C1_MIN) / (C1_MAX - C1_MIN) for score in scores]

def denormalize_c1_scores(normalized_scores):
    """Convert normalized scores back to original C1 range."""
    return [score * (C1_MAX - C1_MIN) + C1_MIN for score in normalized_scores]


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


class EssayDataset(Dataset):
    """Custom Dataset for essay data with C1 labels.

    Args:
        essays: List of essay texts
        labels: List of corresponding C1 labels
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length for tokenization
    """

    def __init__(self, essays: List[str], labels: List[float],
                 tokenizer, max_length: int = 512, normalize_labels: bool = True):
        self.essays = essays

        # Normalize C1 labels for better training stability
        if normalize_labels:
            self.labels = normalize_c1_scores(labels)
        else:
            self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.normalize_labels = normalize_labels

        # Validate inputs
        assert len(essays) == len(labels), "Essays and labels must have same length"

    def __len__(self) -> int:
        return len(self.essays)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        essay = str(self.essays[idx])
        label = float(self.labels[idx])

        # Tokenize the essay with proper error handling
        try:
            encoding = self.tokenizer(
                essay,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
        except Exception as e:
            logger.warning(f"Error tokenizing essay at index {idx}: {e}")
            # Return a fallback encoding for empty/problematic text
            encoding = self.tokenizer(
                "[EMPTY]",
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)  # Use float32 for consistency
        }


class BERTimbauForC1Prediction(nn.Module):
    """BERTimbau model with regression head for C1 prediction.

    Args:
        model_name: HuggingFace model name (e.g., 'neuralmind/bert-base-portuguese-cased')
        num_labels: Number of output labels (1 for regression)
        dropout_prob: Dropout probability for regularization
    """

    def __init__(self, model_name: str, num_labels: int = 1, dropout_prob: float = 0.1):
        super().__init__()  # Modern Python super() syntax
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(self.bert.config.hidden_size, num_labels)

        # Initialize weights with Xavier/Glorot initialization for better training
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
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

        return {
            'loss': loss,
            'logits': logits
        }


def load_and_prepare_data(csv_path: Union[str, pathlib.Path],
                          max_samples: Optional[int] = None) -> Tuple[List[str], List[float]]:
    """Load and prepare the essay dataset.

    Args:
        csv_path: Path to the CSV file containing essays
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        Tuple of (essays, c1_labels)
    """
    logger.info(f"Loading dataset from {csv_path}")
    print(f"[DEBUG] Loading dataset from {csv_path}")

    SAMPLE_SIZE_UPPER_BOUND = 2 ** 31 - 1
    relevant_columns = ["c1", "essay_as_single_utf8_string", "prompt"]
    df = (
        pl.scan_csv(csv_path)
        .head(max_samples if max_samples is not None else SAMPLE_SIZE_UPPER_BOUND)
        .select(relevant_columns)
        .drop_nulls()
        .filter((pl.col("c1") > 0)) # Remove samples with C1 score of 0, as they are not reliable enough
        .unique()
        .collect()
    )

    logger.info(f"Dataset loaded with {len(df)} samples")

    return df


def evaluate_model(model: nn.Module, dataloader: DataLoader,
                   device: torch.device) -> Dict[str, float]:
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            total_loss += outputs['loss'].item()

            # Get normalized predictions and denormalize them for metrics calculation
            normalized_preds = outputs['logits'].squeeze().cpu().numpy()
            denormalized_preds = denormalize_c1_scores(normalized_preds if hasattr(normalized_preds, '__len__') else [normalized_preds])
            predictions.extend(denormalized_preds)

            # Denormalize true labels for metrics calculation
            normalized_labels = labels.cpu().numpy()
            denormalized_labels = denormalize_c1_scores(normalized_labels if hasattr(normalized_labels, '__len__') else [normalized_labels])
            true_labels.extend(denormalized_labels)

    avg_loss = total_loss / len(dataloader)
    mse = mean_squared_error(true_labels, predictions)
    mae = mean_absolute_error(true_labels, predictions)
    r2 = r2_score(true_labels, predictions)

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
        kappa = cohen_kappa_score(true_labels_rounded, predictions_rounded)
    except Exception as e:
        # If kappa calculation fails (e.g., only one class), set to 0
        kappa = 0.0

    # Calculate Quadratic Weighted Kappa
    try:
        qwk = quadratic_weighted_kappa(true_labels_rounded, predictions_rounded, labels=[0, 40, 80, 120, 160, 200])
    except Exception as e:
        qwk = 0.0

    # Calculate Pearson correlation
    try:
        pearson_corr, pearson_p = pearsonr(true_labels, predictions)
        # Handle case where correlation is NaN
        if np.isnan(pearson_corr):
            pearson_corr = 0.0
    except Exception as e:
        pearson_corr = 0.0

    return {
        'loss': avg_loss,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'kappa': kappa,
        'qwk': qwk,
        'pearson_corr': pearson_corr
    }


def train_model(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader,
               device: torch.device, num_epochs: int = 3, learning_rate: float = 2e-5) -> Tuple[List[float], List[Dict[str, float]]]:
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

    # Optimizer and scheduler (using PyTorch's AdamW instead of deprecated transformers version)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * total_steps,
        num_training_steps=total_steps
    )

    logger.info(f"Starting training for {num_epochs} epochs")

    train_losses = []
    val_metrics = []

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        total_train_loss = 0
        train_progress = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")

        for batch in train_progress:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']

            loss.backward()
            # Gradient clipping for stability (modern best practice)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            train_progress.set_postfix({'loss': loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation phase
        val_metrics_epoch = evaluate_model(model, val_dataloader, device)
        val_metrics.append(val_metrics_epoch)

        current_epoch = epoch + 1
        logger.info(f"Epoch {current_epoch} - Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Epoch {current_epoch} - Val Loss: {val_metrics_epoch['loss']:.4f}")
        logger.info(f"Epoch {current_epoch} - Val MSE: {val_metrics_epoch['mse']:.4f}")
        logger.info(f"Epoch {current_epoch} - Val MAE: {val_metrics_epoch['mae']:.4f}")
        logger.info(f"Epoch {current_epoch} - Val R²: {val_metrics_epoch['r2']:.4f}")
        logger.info(f"Epoch {current_epoch} - Val Kappa: {val_metrics_epoch['kappa']:.4f}")
        logger.info(f"Epoch {current_epoch} - Val QWK: {val_metrics_epoch['qwk']:.4f}")
        logger.info(f"Epoch {current_epoch} - Val Pearson: {val_metrics_epoch['pearson_corr']:.4f}")

    return train_losses, val_metrics


def main():
    """Main function to run the fine-tuning process."""
    # Quick device check
    print(f"\n{'='*50}")
    print("BERTimbau C1 Fine-tuning Starting")
    print(f"{'='*50}")

    # Configuration
    MODEL_NAME = "neuralmind/bert-base-portuguese-cased"  # BERTimbau
    MAX_LENGTH = 512  # BERT maximum sequence length
    BATCH_SIZE = 28  # Larger batch size for better GPU utilization
    NUM_EPOCHS = 10  # Fewer epochs may prevent overfitting
    LEARNING_RATE = 3.5e-5  # Scaled learning rate for batch size 28
    MAX_SAMPLES = None  # Set to None to use all samples

    # Check and display device information first
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice Detection:")
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

    # Paths
    project_root = pathlib.Path(__file__).parent.parent
    csv_path = project_root / "generated_datasets" / "extended_essay-br_preprocessed_for_BERT.csv"
    model_save_path = project_root / "models" / "bertimbau_c1_finetuned"

    # Ensure model save directory exists
    model_save_path.mkdir(parents=True, exist_ok=True)

    # Device already configured above

    # Load and prepare data
    dataset = load_and_prepare_data(csv_path, max_samples=MAX_SAMPLES)
    essays = dataset["essay_as_single_utf8_string"]
    c1_labels = dataset["c1"]

    # Create stratification groups for highly imbalanced data
    # Group similar C1 scores to ensure balanced splits
    stratify_groups = [min(label // 200, 4) for label in c1_labels]

    # Split the data with stratification
    train_essays, val_essays, train_labels, val_labels = train_test_split(
        essays, c1_labels, test_size=0.2, random_state=42, stratify=stratify_groups
    )

    total_samples = len(train_essays) + len(val_essays)
    logger.info(f"Training samples: {len(train_essays)} ({len(train_essays) / total_samples:.1%}), Validation samples: {len(val_essays)} ({len(val_essays) / total_samples:.1%}), Total samples: {total_samples}")

    # For regression problems, class weights are not applicable
    # The model will learn to predict continuous C1 scores directly
    logger.info("Using MSE loss for regression - class weights not applicable for continuous targets")
    logger.info(f"Training configuration: BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE:.6f}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create datasets with normalized labels and NLP preprocessing enabled
    train_dataset = EssayDataset(train_essays, train_labels, tokenizer, MAX_LENGTH, normalize_labels=True)
    val_dataset = EssayDataset(val_essays, val_labels, tokenizer, MAX_LENGTH, normalize_labels=True)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = BERTimbauForC1Prediction(MODEL_NAME, num_labels=1)

    # Train the model
    train_losses, val_metrics = train_model(
        model, train_dataloader, val_dataloader, device,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE
    )

    # Save the fine-tuned model with normalization parameters
    logger.info(f"Saving model to {model_save_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'model_name': MODEL_NAME,
        'max_length': MAX_LENGTH,
        'train_losses': train_losses,
        'val_metrics': val_metrics,
        'normalization': {
            'c1_min': C1_MIN,
            'c1_max': C1_MAX,
            'c1_mean': C1_MEAN,
            'c1_std': C1_STD
        }
    }, model_save_path / "bertimbau_c1_model.pth")

    # Save tokenizer separately
    tokenizer.save_pretrained(model_save_path / "tokenizer")

    # Final evaluation
    final_validation_metrics = evaluate_model(model, val_dataloader, device)
    final_validation_metrics_df = pl.DataFrame(final_validation_metrics)

    final_validation_metrics_df.write_csv(model_save_path / "final_validation_metrics.csv")
    print(f"Final validation metrics saved to {model_save_path / 'final_validation_metrics.csv'}")

    final_validation_metrics_df.write_parquet(model_save_path / "final_validation_metrics.parquet")
    print(f"Final validation metrics saved to {model_save_path / 'final_validation_metrics.parquet'}")

    print("Final validation metrics:")
    for metric, value in final_validation_metrics.items():
        print(f"{metric}: {value:.4f}")

    logger.info("Fine-tuning completed successfully!")

    print(f"\n{'='*50}")
    print("✅ Fine-tuning completed successfully!")
    print(f"{'='*50}")
    if torch.cuda.is_available():
        print(f"Trained using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Trained using CPU")
    print(f"Model saved to: {model_save_path}")


if __name__ == "__main__":
    main()
