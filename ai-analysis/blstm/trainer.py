"""Simple but Complete BiLSTM Trainer

This module provides a streamlined Trainer class for the BiLSTM model with:
- Mixed precision training (AMP)
- Learning rate scheduling
- Early stopping
- Gradient clipping
- Comprehensive logging
- Checkpointing

Usage:
    from trainer import BiLSTMTrainer
    from blstm import ModelConfig, TrainConfig

    # Initialize trainer
    trainer = BiLSTMTrainer(model, train_loader, val_loader, configs)

    # Train the model
    best_metrics = trainer.train()
"""

import logging
import time
from pathlib import Path

import torch
import tqdm

# Import from our main implementation
from blstm import (
    BiLSTMRegressor,
    MetricsAccumulator,
    ModelConfig,
    SerializationConfig,
    TargetScaler,
    TrainConfig,
    ensure_dir,
    get_loss_fn,
)
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class BiLSTMTrainer:
    """Simple but complete trainer for BiLSTM regression."""

    def __init__(
        self,
        model: BiLSTMRegressor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_config: ModelConfig,
        train_config: TrainConfig,
        serialization_config: SerializationConfig,
        target_scaler: TargetScaler | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: The BiLSTM model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            model_config: Model configuration
            train_config: Training configuration
            serialization_config: Serialization configuration
            target_scaler: Optional target scaler
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

        # Setup loss function
        self.criterion = get_loss_fn("mae")

        # Training state
        self.current_epoch = 0
        self.best_val_mae = float("inf")
        self.patience_counter = 0
        self.training_history: list[dict[str, float]] = []

        # Ensure output directory exists
        ensure_dir(self.serialization_config.output_dir)

        logger.info(f"Trainer initialized with AMP: {self.use_amp}")

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup the optimizer."""
        if self.train_config.optimizer == "adamw":
            # Use fused AdamW if available (more efficient on CUDA)
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
            # Calculate total steps for OneCycleLR
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

        # Use tqdm progress bar like BERT script
        train_progress = tqdm.tqdm(
            self.train_loader,
            desc=f"Training Epoch {self.current_epoch + 1}",
            leave=False,
        )

        for batch in train_progress:
            # Move data to device
            tokens = batch["tokens"].to(self.device, non_blocking=True)
            lengths = batch["lengths"].to(self.device, non_blocking=True)
            targets = batch["targets"].to(self.device, non_blocking=True)

            # Scale targets if scaler is provided
            if (
                self.target_scaler
                and self.target_scaler.fitted
                and self.target_scaler.mode != "none"
            ):
                targets_scaled = torch.from_numpy(
                    self.target_scaler.transform(targets.cpu().numpy())
                ).to(self.device)
            else:
                targets_scaled = targets

            batch_size = tokens.shape[0]
            total_samples += batch_size

            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with AMP
            if self.use_amp:
                with torch.autocast(
                    "cuda", enabled=True, dtype=self.train_config.amp_dtype
                ):
                    predictions = self.model(tokens, lengths)
                    loss = self.criterion(predictions, targets_scaled)
            else:
                predictions = self.model(tokens, lengths)
                loss = self.criterion(predictions, targets_scaled)

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

            # Update progress bar with current loss like BERT script
            train_progress.set_postfix({"loss": loss.item()})

        return total_loss / total_samples

    def _validate_epoch(self) -> tuple[float, dict[str, float]]:
        """Validate for one epoch and return average loss and metrics."""
        self.model.eval()
        metrics = MetricsAccumulator()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                tokens = batch["tokens"].to(self.device, non_blocking=True)
                lengths = batch["lengths"].to(self.device, non_blocking=True)
                targets = batch["targets"]
                ids = batch["ids"]

                batch_size = tokens.shape[0]
                total_samples += batch_size

                # Forward pass with AMP
                if self.use_amp:
                    with torch.autocast(
                        "cuda", enabled=True, dtype=self.train_config.amp_dtype
                    ):
                        predictions = self.model.predict_and_optionally_clamp(
                            tokens, lengths, clamp_for_metrics=True
                        )
                else:
                    predictions = self.model.predict_and_optionally_clamp(
                        tokens, lengths, clamp_for_metrics=True
                    )

                # Calculate loss for validation monitoring
                if (
                    self.target_scaler
                    and self.target_scaler.fitted
                    and self.target_scaler.mode != "none"
                ):
                    targets_scaled = torch.from_numpy(
                        self.target_scaler.transform(targets.numpy())
                    )
                    preds_scaled = torch.from_numpy(
                        self.target_scaler.transform(predictions.cpu().numpy())
                    )
                    loss = self.criterion(preds_scaled, targets_scaled)
                else:
                    loss = self.criterion(predictions.cpu(), targets)

                total_loss += loss.item() * batch_size

                # Update metrics (predictions are already clamped and on original scale)
                metrics.update(predictions.cpu(), targets, ids)

        # Compute metrics
        computed_metrics = metrics.compute_metrics(self.target_scaler)
        avg_loss = total_loss / total_samples

        return avg_loss, computed_metrics

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
        logger.info(f"Starting training for {self.train_config.epochs} epochs")
        start_time = time.time()

        best_metrics = {}

        try:
            for epoch in range(self.train_config.epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()

                # Training phase
                train_loss = self._train_epoch()

                # Validation phase
                val_loss, val_metrics = self._validate_epoch()

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
                    "val_rmse": val_mae,
                    "val_mae": val_metrics.get("mae", 0.0),
                    "val_step_accuracy": val_metrics.get("step_accuracy", 0.0),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "epoch_time": epoch_time,
                }
                self.training_history.append(history_entry)

                # BERT-style logging
                current_epoch = epoch + 1
                logger.info(f"Epoch {current_epoch}/{self.train_config.epochs}")
                logger.info(f"Epoch {current_epoch} - Train Loss: {train_loss:.4f}")
                logger.info(
                    f"Epoch {current_epoch} - Val Loss: {val_metrics['loss']:.4f}"
                )
                logger.info(
                    f"Epoch {current_epoch} - Val MSE: {val_metrics['mse']:.4f}"
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
            logger.info(f"Training completed in {total_time:.1f}s")
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


def create_trainer_from_configs(
    model: BiLSTMRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_config: ModelConfig,
    train_config: TrainConfig,
    target_scaler: TargetScaler | None = None,
    output_dir: str = "runs/bilstm",
) -> BiLSTMTrainer:
    """Convenience function to create trainer from configs."""
    serialization_config = SerializationConfig(output_dir=output_dir)

    return BiLSTMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_config=model_config,
        train_config=train_config,
        serialization_config=serialization_config,
        target_scaler=target_scaler,
    )


# # Example usage function
# def example_training_workflow() -> tuple[BiLSTMTrainer, dict[str, float]]:
#     """Example of how to use the BiLSTMTrainer."""
#     import numpy as np
#     from blstm import (
#         EssayDataset,
#         TargetScaler,
#         collate_batch,
#         get_device,
#         set_seed,
#         split_dataset,
#     )
#     from torch.utils.data import DataLoader
#
#     # Setup
#     device = get_device("auto")
#     set_seed(42)
#
#     # Create synthetic dataset (replace with your real data)
#     arrays = []
#     scores = []
#     valid_scores = [0, 40, 80, 120, 160, 200]
#
#     for i in range(1000):  # Larger dataset for better training
#         seq_len = np.random.randint(10, 500)
#         embedding = np.random.randn(seq_len, 768).astype(np.float32)
#         score = float(np.random.choice(valid_scores))
#         arrays.append(embedding)
#         scores.append(score)
#
#     dataset = EssayDataset(pl.DataFrame(arrays, scores))
#     train_dataset, val_dataset, _ = split_dataset(dataset, 0.2, 0.0, seed=42)
#
#     # Create data loaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=32,
#         shuffle=True,
#         collate_fn=lambda batch: collate_batch(batch, 0.0),
#         num_workers=2,
#         pin_memory=True,
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=32,
#         shuffle=False,
#         collate_fn=lambda batch: collate_batch(batch, 0.0),
#         num_workers=2,
#         pin_memory=True,
#     )
#
#     # Setup target scaler
#     all_targets = [record.score for record in dataset.records]
#     target_scaler = TargetScaler("minmax")
#     target_scaler.fit(np.array(all_targets))
#
#     # Create configs
#     model_config = ModelConfig(
#         hidden_size=256, num_layers=2, dropout=0.1, aggregation="last"
#     )
#
#     train_config = TrainConfig(
#         epochs=10,
#         batch_size=32,
#         lr=2e-4,
#         scheduler="plateau",
#         early_stopping_patience=3,
#         device="auto",
#     )
#
#     # Create model
#     model = BiLSTMRegressor(model_config)
#
#     # Create trainer
#     trainer = create_trainer_from_configs(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         model_config=model_config,
#         train_config=train_config,
#         target_scaler=target_scaler,
#         output_dir="runs/example_training",
#     )
#
#     # Train the model
#     best_metrics = trainer.train()
#
#     logger.info(f"Training completed! Best metrics: {best_metrics}")
#     return trainer, best_metrics


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # # Run example
    # trainer, metrics = example_training_workflow()
    # print(f"Final metrics: {metrics}")
