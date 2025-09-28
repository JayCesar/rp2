#!/usr/bin/env python3
"""
Simple Training Example for BiLSTM

This script demonstrates how to use the BiLSTMTrainer for essay C1 score prediction.
It shows the complete workflow from data loading to model training.
"""

import logging
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Import our modules
from blstm import (
    BiLSTMRegressor, ModelConfig, TrainConfig, SerializationConfig,
    EmbeddingSequenceDataset, collate_batch, split_dataset,
    TargetScaler, get_device, set_seed
)
from trainer import BiLSTMTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data(n_samples: int = 500) -> EmbeddingSequenceDataset:
    """Create sample dataset for demonstration."""
    logger.info(f"Creating sample dataset with {n_samples} examples...")
    
    arrays = []
    scores = []
    valid_scores = [0, 40, 80, 120, 160, 200]  # C1 score levels
    
    for i in range(n_samples):
        # Variable sequence length between 20 and 300 tokens
        seq_len = np.random.randint(20, 301)
        
        # Random 768-dimensional embeddings (replace with real BERT embeddings)
        embedding = np.random.randn(seq_len, 768).astype(np.float32)
        
        # Random C1 score
        score = float(np.random.choice(valid_scores))
        
        arrays.append(embedding)
        scores.append(score)
    
    return EmbeddingSequenceDataset.from_memory(arrays, scores)


def main():
    """Main training workflow."""
    logger.info("Starting BiLSTM training example...")
    
    # 1. Setup device and reproducibility
    device = get_device('auto')
    set_seed(42)
    logger.info(f"Using device: {device}")
    
    # 2. Create or load dataset
    dataset = create_sample_data(n_samples=800)
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    # 3. Split dataset
    train_dataset, val_dataset, _ = split_dataset(
        dataset, val_ratio=0.2, test_ratio=0.0, seed=42
    )
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # 4. Create data loaders
    batch_size = 16  # Small batch size for demo
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, 0.0),
        num_workers=2, 
        pin_memory=device.type == 'cuda'
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, 0.0),
        num_workers=2, 
        pin_memory=device.type == 'cuda'
    )
    
    # 5. Setup target scaler
    all_targets = [record.score for record in dataset.records]
    target_scaler = TargetScaler('minmax')
    target_scaler.fit(np.array(all_targets))
    logger.info(f"Target scaler fitted: {target_scaler.mode}")
    
    # 6. Create model configuration
    model_config = ModelConfig(
        input_dim=768,
        hidden_size=128,        # Smaller for faster training
        num_layers=2,
        bidirectional=True,
        dropout=0.2,
        aggregation='last',     # Try 'mean', 'max', or 'attn' too
        mlp_hidden=128,
        use_layer_norm=True
    )
    
    # 7. Create training configuration
    train_config = TrainConfig(
        epochs=15,
        batch_size=batch_size,
        lr=1e-3,               # Higher learning rate for faster convergence
        weight_decay=1e-4,
        optimizer='adamw',
        scheduler='plateau',    # Will reduce LR when validation plateaus
        plateau_patience=3,
        plateau_factor=0.5,
        early_stopping_patience=5,
        grad_clip_norm=1.0,
        use_amp=True,          # Mixed precision for faster training
        amp_dtype='bf16',      # bfloat16 is more stable than fp16
        target_scaler='minmax'
    )
    
    # 8. Create serialization configuration
    output_dir = 'runs/bilstm_example'
    serialization_config = SerializationConfig(
        output_dir=output_dir,
        save_best_only=True,
        keep_last_k=3
    )
    
    # 9. Create model
    model = BiLSTMRegressor(model_config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 10. Create trainer
    trainer = BiLSTMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_config=model_config,
        train_config=train_config,
        serialization_config=serialization_config,
        target_scaler=target_scaler,
        device=device
    )
    
    # 11. Train the model
    logger.info("Starting training...")
    try:
        best_metrics = trainer.train()
        logger.info("Training completed successfully!")
        logger.info(f"Best validation metrics: {best_metrics}")
        
        # 12. Save training summary
        summary_path = Path(output_dir) / 'training_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("BiLSTM Training Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Configuration:\n{model_config.to_dict()}\n\n")
            f.write(f"Training Configuration:\n{train_config.to_dict()}\n\n")
            f.write(f"Best Validation Metrics:\n{best_metrics}\n\n")
            f.write(f"Training History (last 5 epochs):\n")
            for entry in trainer.training_history[-5:]:
                f.write(f"  Epoch {entry['epoch']}: "
                        f"Train Loss={entry['train_loss']:.6f}, "
                        f"Val RMSE={entry['val_rmse']:.2f}, "
                        f"Val Step Acc={entry['val_step_accuracy']:.3f}, "
                        f"Time={entry['epoch_time']:.1f}s\n")
        
        logger.info(f"Training summary saved to: {summary_path}")
        
        # 13. Example of loading the trained model
        logger.info("Example: Loading best checkpoint...")
        best_checkpoint = Path(output_dir) / 'best.pt'
        if best_checkpoint.exists():
            # Create a new trainer instance (simulation of loading in new session)
            new_trainer = BiLSTMTrainer(
                model=BiLSTMRegressor(model_config),
                train_loader=train_loader,
                val_loader=val_loader,
                model_config=model_config,
                train_config=train_config,
                serialization_config=serialization_config,
                target_scaler=target_scaler,
                device=device
            )
            new_trainer.load_checkpoint(str(best_checkpoint))
            logger.info("Checkpoint loaded successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())