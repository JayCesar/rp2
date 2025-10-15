#!/usr/bin/env python3
"""Test BiLSTM Training with Real Embeddings"""

import logging
import sys
import os

sys.path.append("..")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Test BiLSTM training with real embeddings"""
    try:
        # Import our modules
        from blstm_training import (
            BiLSTMRegressor,
            ModelConfig,
            TrainConfig,
            SerializationConfig,
            EmbeddingSequenceDataset,
            collate_batch,
            split_dataset,
            TargetScaler,
            get_device,
            set_seed,
        )
        from trainer import BiLSTMTrainer
        from torch.utils.data import DataLoader

        logger.info("Starting BiLSTM training test with real embeddings...")

        # 1. Setup device and reproducibility
        device = get_device("auto")
        set_seed(42)
        logger.info(f"Using device: {device}")

        # 2. Load dataset from CSV with embeddings
        csv_path = "../embeddings_test/embeddings_metadata.csv"
        if not os.path.exists(csv_path):
            logger.error(f"Embeddings metadata CSV not found at {csv_path}")
            return 1

        dataset = EmbeddingSequenceDataset.from_csv(
            csv_path=csv_path,
            id_column="id",
            embedding_column="embedding_path",
            score_column="c1",
            embedding_format="npy",
        )
        logger.info(f"Loaded dataset with {len(dataset)} samples")

        # 3. Split dataset (80% train, 20% val)
        train_dataset, val_dataset, _ = split_dataset(
            dataset, val_ratio=0.2, test_ratio=0.0, seed=42
        )
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        # 4. Create data loaders
        batch_size = 8  # Small batch size for demo
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_batch(batch, 0.0),
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, 0.0),
            num_workers=0,
            pin_memory=False,
        )

        # 5. Setup target scaler
        all_targets = [record.score for record in dataset.records]
        target_scaler = TargetScaler("minmax")
        target_scaler.fit(np.array(all_targets))
        logger.info(f"Target scaler fitted: {target_scaler.mode}")

        # 6. Create model configuration (small for quick testing)
        model_config = ModelConfig(
            input_dim=768,
            hidden_size=64,  # Small for faster training
            num_layers=1,  # Single layer for simplicity
            bidirectional=True,
            dropout=0.1,
            aggregation="last",
            mlp_hidden=64,
            use_layer_norm=False,
        )

        # 7. Create training configuration (short training for testing)
        train_config = TrainConfig(
            epochs=3,  # Just 3 epochs for testing
            batch_size=batch_size,
            lr=1e-3,
            weight_decay=1e-4,
            optimizer="adamw",
            scheduler="none",  # No scheduler for simple test
            early_stopping_patience=10,
            grad_clip_norm=1.0,
            use_amp=False,  # Disable AMP for simplicity
            target_scaler="minmax",
        )

        # 8. Create serialization configuration
        output_dir = "runs/test_training"
        serialization_config = SerializationConfig(
            output_dir=output_dir, save_best_only=True, keep_last_k=2
        )

        # 9. Create model
        model = BiLSTMRegressor(model_config)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created with {total_params:,} parameters")

        # 10. Create trainer
        trainer = BiLSTMTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_config=model_config,
            train_config=train_config,
            serialization_config=serialization_config,
            target_scaler=target_scaler,
            device=device,
        )

        # 11. Train the model
        logger.info("Starting training...")
        best_metrics = trainer.train()
        logger.info("Training completed successfully!")
        logger.info(f"Best validation metrics: {best_metrics}")

        # 12. Test inference
        logger.info("Testing inference...")
        from blstm import evaluate_model

        # Load best model for inference test
        checkpoint_path = f"{output_dir}/best.pt"
        if os.path.exists(checkpoint_path):
            trainer.load_checkpoint(checkpoint_path)
            logger.info("Best checkpoint loaded for inference test")

            # Run evaluation
            metrics, predictions = evaluate_model(
                model, val_loader, device, target_scaler
            )
            logger.info(f"Inference test metrics: {metrics}")
            logger.info(f"Sample predictions: {predictions[:3]}")

        logger.info("Complete test successful!")
        return 0

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import numpy as np

    exit(main())

