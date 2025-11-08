"""Model-Agnostic Trainer for Essay Score Regression

This module provides a clean, reusable Trainer class that works with any regression model.
Supports mixed precision training, early stopping, and comprehensive metrics logging.

Key features:
- AdamW optimizer with configurable learning rate and weight decay
- Mixed precision training (AMP) for faster GPU training
- Early stopping based on validation MAE
- Comprehensive metrics: MAE, RMSE, R², Kappa, QWK, Pearson correlation
- Checkpoint saving (best and latest models)
- Training history tracking

Usage:
    from trainer import Trainer
    from conv1d import Conv1DRegressor, ModelConfig, TrainConfig
    
    trainer = Trainer(model, train_loader, val_loader, configs)
    best_metrics = trainer.train()
"""

import pathlib
import time

import polars as pl
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

# Import utilities using relative imports
try:
    from ..feature_extraction.utils import logger
    from ..blstm.blstm import MetricsAccumulator, TargetScaler, ensure_dir
    from .conv1d import ModelConfig, SerializationConfig, TrainConfig
except ImportError:
    # Fallback for direct script execution
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from feature_extraction.utils import logger
    from blstm.blstm import MetricsAccumulator, TargetScaler, ensure_dir
    from conv1d import ModelConfig, SerializationConfig, TrainConfig


class Trainer:
    """Model-agnostic trainer for regression models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_config: ModelConfig,
        train_config: TrainConfig,
        serialization_config: SerializationConfig,
        target_scaler: TargetScaler | None = None,
        device: torch.device | None = None,
    ):
        """Initialize the trainer.
        
        Args:
            model: Neural network model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            model_config: Model configuration
            train_config: Training configuration
            serialization_config: Checkpointing configuration
            target_scaler: Optional target scaler for normalization
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
        
        # Setup mixed precision
        self.use_amp = train_config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        
        # Setup loss function
        self.criterion = nn.L1Loss()  # MAE loss
        
        # Training state
        self.current_epoch = 0
        self.best_val_mae = float("inf")
        self.patience_counter = 0
        self.training_history = []
        
        # Ensure output directory exists
        ensure_dir(self.serialization_config.output_dir)
        
        logger.info(f"Trainer initialized with AMP: {self.use_amp}")
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup AdamW optimizer."""
        if self.train_config.optimizer == "adamw":
            # Use fused AdamW if available (more efficient on CUDA)
            optimizer_kwargs = {
                "lr": self.train_config.lr,
                "weight_decay": self.train_config.weight_decay,
            }
            
            if self.device.type == "cuda":
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
    
    def _train_epoch(self) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        # Progress bar
        train_progress = tqdm.tqdm(
            self.train_loader,
            desc=f"Training Epoch {self.current_epoch + 1}",
            leave=False,
        )
        
        for batch in train_progress:
            # Move data to device
            tokens = batch["tokens"].to(self.device, non_blocking=True)
            lengths = batch["lengths"].to(self.device, non_blocking=True) if batch["lengths"] is not None else None
            targets = batch["targets"].to(self.device, non_blocking=True)
            
            # Scale targets if needed
            if self.target_scaler and self.target_scaler.fitted and self.target_scaler.mode != "none":
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
                with torch.autocast("cuda",enabled=True, dtype=self.train_config.amp_dtype):
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
            
            # Update progress bar
            train_progress.set_postfix({"loss": loss.item()})
        
        return total_loss / total_samples
    
    def _validate_epoch(self) -> tuple[float, dict[str, float]]:
        """Validate for one epoch and return loss and metrics."""
        self.model.eval()
        metrics = MetricsAccumulator()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                tokens = batch["tokens"].to(self.device, non_blocking=True)
                lengths = batch["lengths"].to(self.device, non_blocking=True) if batch["lengths"] is not None else None
                targets = batch["targets"]
                ids = batch["ids"]
                
                batch_size = tokens.shape[0]
                total_samples += batch_size
                
                # Forward pass with AMP
                if self.use_amp:
                    with torch.autocast("cuda",enabled=True, dtype=self.train_config.amp_dtype):
                        predictions = self.model(tokens, lengths)
                else:
                    predictions = self.model(tokens, lengths)
                
                # Clamp predictions for metrics
                min_val, max_val = self.model_config.output_range
                predictions_clamped = torch.clamp(predictions, min_val, max_val)
                
                # Calculate loss
                if self.target_scaler and self.target_scaler.fitted and self.target_scaler.mode != "none":
                    targets_scaled = torch.from_numpy(
                        self.target_scaler.transform(targets.numpy())
                    )
                    preds_scaled = torch.from_numpy(
                        self.target_scaler.transform(predictions_clamped.cpu().numpy())
                    )
                    loss = self.criterion(preds_scaled, targets_scaled)
                else:
                    loss = self.criterion(predictions_clamped.cpu(), targets)
                
                total_loss += loss.item() * batch_size
                
                # Update metrics (on original scale)
                metrics.update(predictions_clamped.cpu().float(), targets, ids)
        
        # Compute metrics
        computed_metrics = metrics.compute_metrics(self.target_scaler)
        avg_loss = total_loss / total_samples
        
        return avg_loss, computed_metrics
    
    def _save_checkpoint(self, is_best: bool = False, metrics: dict[str, float] | None = None):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "model_config": self.model_config.to_dict(),
            "train_config": self.train_config.to_dict(),
            "target_scaler_state": self.target_scaler.state_dict() if self.target_scaler else None,
            "best_val_mae": self.best_val_mae,
            "training_history": self.training_history,
            "metrics": metrics or {},
        }
        
        # Save latest checkpoint
        latest_path = pathlib.Path(self.serialization_config.output_dir) / "latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = pathlib.Path(self.serialization_config.output_dir) / "best.pt"
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
                logger.info(f"Epoch {current_epoch} - Val Loss: {val_metrics['loss']:.4f}")
                logger.info(f"Epoch {current_epoch} - Val MAE: {val_metrics['mae']:.4f}")
                logger.info(f"Epoch {current_epoch} - Val RMSE: {val_metrics['rmse']:.4f}")
                logger.info(f"Epoch {current_epoch} - Val R²: {val_metrics['r2']:.4f}")
                logger.info(f"Epoch {current_epoch} - Val Kappa: {val_metrics['kappa']:.4f}")
                logger.info(f"Epoch {current_epoch} - Val QWK: {val_metrics['qwk']:.4f}")
                logger.info(f"Epoch {current_epoch} - Val Pearson: {val_metrics['pearson_corr']:.4f}")
                logger.info(f"Epoch {current_epoch} - Val Step Acc: {val_metrics.get('step_accuracy', 0.0):.4f}")
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
