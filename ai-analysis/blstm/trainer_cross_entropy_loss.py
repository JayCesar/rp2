"""BiLSTM Trainer with CrossEntropy Loss for Classification

This module provides a Trainer class for BiLSTM classification models using
CrossEntropyLoss on 6-class C1 score classification {0, 40, 80, 120, 160, 200}.

Mirrors the structure of trainer.py but adapted for classification:
- Uses BiLSTMClassifier instead of BiLSTMRegressor
- Computes CrossEntropyLoss on class indices
- Converts predictions back to score space for metrics
- Tracks best model by validation MAE (like regression)

Usage:
    from trainer_cross_entropy_loss import BiLSTMCETrainer
    from blstm import ModelConfig, TrainConfig
    from blstm_cross_entropy_loss import BiLSTMClassifier

    trainer = BiLSTMCETrainer(model, train_loader, val_loader, configs)
    best_metrics = trainer.train()
"""

import logging
import time
from pathlib import Path

import torch
import tqdm

# Import from main implementation
from blstm import (
    MetricsAccumulator,
    ModelConfig,
    SerializationConfig,
    TargetScaler,
    TrainConfig,
    ensure_dir,
)
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

# Import CE-specific components
from blstm_cross_entropy_loss import (
    BiLSTMClassifier,
    scores_to_class_indices,
    logits_to_scores,
)

logger = logging.getLogger(__name__)


class BiLSTMCETrainer:
    """Trainer for BiLSTM classification with CrossEntropyLoss."""

    def __init__(
        self,
        model: BiLSTMClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_config: ModelConfig,
        train_config: TrainConfig,
        serialization_config: SerializationConfig,
        target_scaler: TargetScaler | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the CE trainer.

        Args:
            model: The BiLSTMClassifier to train
            train_loader: Training data loader
            val_loader: Validation data loader
            model_config: Model configuration
            train_config: Training configuration
            serialization_config: Serialization configuration
            target_scaler: Optional target scaler (not used for CE but kept for API parity)
            device: Device to train on (auto-detected if None)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_config = model_config
        self.train_config = train_config
        self.serialization_config = serialization_config
        self.target_scaler = target_scaler

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = self.model.to(self.device)
        logger.info(f"Training on device: {self.device}")

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Setup scheduler
        self.scheduler = self._setup_scheduler()

        # Setup mixed precision
        self.use_amp = train_config.use_amp and self.device.type == "cuda"
        self.scaler = torch.GradScaler("cuda", enabled=self.use_amp)

        # Setup loss function - CrossEntropyLoss
        self.criterion = torch.nn.CrossEntropyLoss()
        logger.info("Using CrossEntropyLoss for 6-class classification")

        # Training state
        self.current_epoch = 0
        self.best_val_mae = float("inf")
        self.patience_counter = 0
        self.training_history: list[dict[str, float]] = []
        self.best_val_predictions: list[dict[str, str | float | int]] = []

        # Ensure output directory exists
        ensure_dir(self.serialization_config.output_dir)

        logger.info(f"CE Trainer initialized with AMP: {self.use_amp}")

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup the optimizer."""
        if self.train_config.optimizer == "adamw":
            use_fused = self.device.type == "cuda" and hasattr(
                torch.optim.AdamW, "__init__"
            )
            optimizer_kwargs = {
                "lr": self.train_config.lr,
                "weight_decay": self.train_config.weight_decay,
            }
            if use_fused:
                try:
                    optimizer = torch.optim.AdamW(
                        self.model.parameters(), fused=True, **optimizer_kwargs
                    )
                    logger.info("Using fused AdamW optimizer")
                except:
                    optimizer = torch.optim.AdamW(
                        self.model.parameters(), **optimizer_kwargs
                    )
                    logger.info("Using standard AdamW optimizer")
            else:
                optimizer = torch.optim.AdamW(
                    self.model.parameters(), **optimizer_kwargs
                )
                logger.info("Using standard AdamW optimizer")
            return optimizer
        else:
            raise ValueError(f"Unknown optimizer: {self.train_config.optimizer}")

    def _setup_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Setup the learning rate scheduler."""
        if self.train_config.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.train_config.plateau_factor,
                patience=self.train_config.plateau_patience,
            )
            logger.info("Using ReduceLROnPlateau scheduler")
            return scheduler
        if self.train_config.scheduler == "onecycle":
            total_steps = len(self.train_loader) * self.train_config.epochs
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.train_config.lr,
                total_steps=total_steps,
                pct_start=self.train_config.onecycle_pct_start,
            )
            logger.info(f"Using OneCycleLR scheduler (total_steps: {total_steps})")
            return scheduler
        if self.train_config.scheduler == "none":
            logger.info("No scheduler used")
            return None

        raise ValueError(f"Unknown scheduler: {self.train_config.scheduler}")

    def _train_epoch(self) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        train_progress = tqdm.tqdm(
            self.train_loader,
            desc=f"Training Epoch {self.current_epoch + 1}",
            leave=False,
        )

        for batch in train_progress:
            # Move data to device
            tokens = batch["tokens"].to(self.device, non_blocking=True)
            lengths = batch["lengths"]
            targets = batch["targets"].to(self.device, non_blocking=True)
            
            # Handle feature mode (lengths is None)
            if lengths is None:
                # Feature mode: reshape [B, F] -> [B, 1, F] with length=1 for each sample
                tokens = tokens.unsqueeze(1)  # Add sequence dimension
                lengths = torch.ones(tokens.shape[0], dtype=torch.long, device=self.device)
            else:
                lengths = lengths.to(self.device, non_blocking=True)

            # Convert scores to class indices for CE loss
            target_classes = scores_to_class_indices(targets).long()

            batch_size = tokens.shape[0]
            total_samples += batch_size

            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with AMP
            if self.use_amp:
                with torch.autocast(
                    "cuda", enabled=True, dtype=self.train_config.amp_dtype
                ):
                    logits = self.model(tokens, lengths)
                    loss = self.criterion(logits, target_classes)
            else:
                logits = self.model(tokens, lengths)
                loss = self.criterion(logits, target_classes)

            # Backward pass with AMP
            if self.use_amp:
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.train_config.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.train_config.grad_clip_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                if self.train_config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.train_config.grad_clip_norm
                    )

                self.optimizer.step()

            total_loss += loss.item() * batch_size

            # Update OneCycleLR scheduler
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            train_progress.set_postfix({"loss": loss.item()})

        return total_loss / total_samples

    def _validate_epoch(self) -> tuple[float, dict[str, float], list[dict[str, str | float | int]]]:
        """Validate for one epoch and return average loss, metrics, and predictions."""
        self.model.eval()
        metrics = MetricsAccumulator()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                tokens = batch["tokens"].to(self.device, non_blocking=True)
                lengths = batch["lengths"]
                targets = batch["targets"]
                ids = batch["ids"]
                
                # Handle feature mode (lengths is None)
                if lengths is None:
                    # Feature mode: reshape [B, F] -> [B, 1, F] with length=1 for each sample
                    tokens = tokens.unsqueeze(1)  # Add sequence dimension
                    lengths = torch.ones(tokens.shape[0], dtype=torch.long, device=self.device)
                else:
                    lengths = lengths.to(self.device, non_blocking=True)

                batch_size = tokens.shape[0]
                total_samples += batch_size

                # Forward pass with AMP
                if self.use_amp:
                    with torch.autocast(
                        "cuda", enabled=True, dtype=self.train_config.amp_dtype
                    ):
                        logits = self.model(tokens, lengths)
                        predictions = logits_to_scores(logits)
                else:
                    logits = self.model(tokens, lengths)
                    predictions = logits_to_scores(logits)

                # Calculate loss for validation monitoring
                target_classes = scores_to_class_indices(targets.to(self.device)).long()
                loss = self.criterion(logits, target_classes)
                total_loss += loss.item() * batch_size

                # Update metrics (predictions already in score space {0, 40, 80, 120, 160, 200})
                metrics.update(predictions.cpu(), targets, ids)

        # Compute metrics and per-sample predictions
        computed_metrics = metrics.compute_metrics(self.target_scaler)
        predictions = metrics.get_predictions_df(self.target_scaler)
        avg_loss = total_loss / total_samples

        return avg_loss, computed_metrics, predictions

    def _save_checkpoint(
        self, is_best: bool = False, metrics: dict[str, float] | None = None
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "model_config": self.model_config.to_dict(),
            "train_config": self.train_config.to_dict(),
            "target_scaler_state": self.target_scaler.state_dict()
            if self.target_scaler
            else None,
            "best_val_mae": self.best_val_mae,
            "training_history": self.training_history,
            "metrics": metrics or {},
            "loss_type": "CrossEntropyLoss",
        }

        # Save latest checkpoint
        latest_path = Path(self.serialization_config.output_dir) / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = Path(self.serialization_config.output_dir) / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with val_mae: {self.best_val_mae:.6f}")

        logger.info(f"Checkpoint saved: {latest_path}")

    def train(self) -> dict[str, float]:
        """Train the model and return best validation metrics."""
        logger.info(f"Starting CE training for {self.train_config.epochs} epochs")
        start_time = time.time()

        best_metrics = {}

        try:
            for epoch in range(self.train_config.epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()

                # Training phase
                train_loss = self._train_epoch()

                # Validation phase
                val_loss, val_metrics, val_predictions = self._validate_epoch()

                # Update scheduler (except OneCycleLR which updates per step)
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["mae"])

                # Check for improvement
                val_mae = val_metrics["mae"]
                is_best = val_mae < self.best_val_mae

                if is_best:
                    self.best_val_mae = val_mae
                    self.patience_counter = 0
                    best_metrics = val_metrics.copy()
                    self.best_val_predictions = val_predictions
                else:
                    self.patience_counter += 1

                # Save checkpoint
                self._save_checkpoint(is_best=is_best, metrics=val_metrics)

                # Update training history
                epoch_time = time.time() - epoch_start_time
                history_entry = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "val_rmse": val_metrics["rmse"],
                    "kappa": val_metrics["kappa"],
                    "qwk": val_metrics["qwk"],
                    "r2": val_metrics["r2"],
                    "pearson_corr": val_metrics["pearson_corr"],
                    "step_accuracy": val_metrics["step_accuracy"],
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "epoch_time": epoch_time,
                }
                self.training_history.append(history_entry)

                # Logging
                current_epoch = epoch + 1
                logger.info(f"Epoch {current_epoch}/{self.train_config.epochs}")
                logger.info(f"Epoch {current_epoch} - Train Loss: {train_loss:.4f}")
                logger.info(
                    f"Epoch {current_epoch} - Val Loss: {val_metrics['loss']:.4f}"
                )
                logger.info(
                    f"Epoch {current_epoch} - Val MAE: {val_metrics['mae']:.4f}"
                )
                logger.info(
                    f"Epoch {current_epoch} - Val RMSE: {val_metrics['rmse']:.4f}"
                )
                logger.info(f"Epoch {current_epoch} - Val R²: {val_metrics['r2']:.4f}")
                logger.info(
                    f"Epoch {current_epoch} - Val Kappa: {val_metrics['kappa']:.4f}"
                )
                logger.info(
                    f"Epoch {current_epoch} - Val QWK: {val_metrics['qwk']:.4f}"
                )
                logger.info(
                    f"Epoch {current_epoch} - Val Pearson: {val_metrics['pearson_corr']:.4f}"
                )
                logger.info(
                    f"Epoch {current_epoch} - Val Step Acc: {val_metrics.get('step_accuracy', 0.0):.4f}"
                )
                logger.info(f"Epoch {current_epoch} - Time: {epoch_time:.1f}s")

                # Early stopping check
                if self.patience_counter >= self.train_config.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs "
                        f"(patience: {self.patience_counter}/{self.train_config.early_stopping_patience})"
                    )
                    break

            total_time = time.time() - start_time
            logger.info(f"CE training completed in {total_time:.1f}s")
            logger.info(f"Best validation MAE: {self.best_val_mae:.6f}")

            return best_metrics

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            return best_metrics
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load scaler state
        if "scaler_state_dict" in checkpoint and self.use_amp:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Load target scaler state
        if "target_scaler_state" in checkpoint and self.target_scaler:
            self.target_scaler.load_state_dict(checkpoint["target_scaler_state"])

        # Load training state
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_mae = checkpoint.get("best_val_mae", float("inf"))
        self.training_history = checkpoint.get("training_history", [])

        logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")


def create_ce_trainer_from_configs(
    model: BiLSTMClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_config: ModelConfig,
    train_config: TrainConfig,
    target_scaler: TargetScaler | None = None,
    output_dir: str = "runs/bilstm_ce",
) -> BiLSTMCETrainer:
    """Convenience function to create CE trainer from configs."""
    serialization_config = SerializationConfig(output_dir=output_dir)

    return BiLSTMCETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_config=model_config,
        train_config=train_config,
        serialization_config=serialization_config,
        target_scaler=target_scaler,
    )


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
