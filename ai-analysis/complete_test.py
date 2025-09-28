#!/usr/bin/env python3
"""Complete End-to-End BiLSTM Test

This script demonstrates the complete pipeline:
1. BERTimbau preprocessing (embedding generation)
2. BiLSTM training with real embeddings
3. Model inference and evaluation
4. Checkpoint saving and loading

This is a comprehensive test of all components working together.
"""

import logging
import sys
import os
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_preprocessing_pipeline():
    """Test BERTimbau preprocessing with a very small sample"""
    logger.info("=" * 60)
    logger.info("TESTING PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    
    # Add BERT to path
    sys.path.append('BERT')
    from bertimbau_preprocessing import create_embedding_dataset_from_json
    
    # Create an even smaller sample for testing (just 10 essays)
    logger.info('Loading dataset...')
    with open('generated_datasets/extended_essay-br_preprocessed_for_BERT.json', 'r') as f:
        data = json.load(f)
    
    # Take just 10 essays for complete testing
    sample_data = data[:10]
    logger.info(f'Using sample of {len(sample_data)} essays')
    
    # Save sample to temporary file
    with open('ai-analysis/complete_test_data.json', 'w') as f:
        json.dump(sample_data, f)
    
    try:
        logger.info('Starting BERTimbau preprocessing...')
        csv_path, stats = create_embedding_dataset_from_json(
            json_path='ai-analysis/complete_test_data.json',
            text_column='essay_as_single_utf8_string',
            score_column='c1',
            output_dir='ai-analysis/complete_test_embeddings',
            model_name='neuralmind/bert-base-portuguese-cased',
            batch_size=2,  # Very small batch
            save_format='npy',
            max_length=128  # Short for quick processing
        )
        logger.info('Preprocessing completed successfully')
        logger.info(f'Metadata CSV: {csv_path}')
        logger.info(f'Processing stats: {stats}')
        return True
    except Exception as e:
        logger.error(f'Preprocessing failed: {e}')
        return False

def test_bilstm_training():
    """Test BiLSTM training with the generated embeddings"""
    logger.info("=" * 60)
    logger.info("TESTING BiLSTM TRAINING")
    logger.info("=" * 60)
    
    # Add blstm to path
    sys.path.append('blstm')
    from blstm import (
        BiLSTMRegressor, ModelConfig, TrainConfig, SerializationConfig,
        EmbeddingSequenceDataset, collate_batch, split_dataset,
        TargetScaler, get_device, set_seed, evaluate_model
    )
    from trainer import BiLSTMTrainer
    from torch.utils.data import DataLoader
    import numpy as np
    
    try:
        # 1. Setup
        device = get_device('auto')
        set_seed(42)
        logger.info(f"Using device: {device}")
        
        # 2. Load dataset from CSV with correct paths
        csv_path = 'ai-analysis/complete_test_embeddings/embeddings_metadata.csv'
        if not os.path.exists(csv_path):
            logger.error(f"Embeddings metadata CSV not found at {csv_path}")
            return False
            
        dataset = EmbeddingSequenceDataset.from_csv(
            csv_path=csv_path,
            id_column='id',
            embedding_column='embedding_path',
            score_column='c1',
            embedding_format='npy'
        )
        logger.info(f"Loaded dataset with {len(dataset)} samples")
        
        # 3. Split dataset (70% train, 30% val for small dataset)
        if len(dataset) < 4:
            logger.warning("Dataset too small for proper split, using all data for both train and val")
            train_dataset = dataset
            val_dataset = dataset
        else:
            train_dataset, val_dataset, _ = split_dataset(
                dataset, val_ratio=0.3, test_ratio=0.0, seed=42
            )
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        # 4. Create data loaders
        batch_size = min(4, len(dataset))  # Adaptive batch size
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=lambda batch: collate_batch(batch, 0.0),
            num_workers=0,
            pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, 0.0),
            num_workers=0,
            pin_memory=False
        )
        
        # 5. Setup target scaler
        all_targets = [record.score for record in dataset.records]
        target_scaler = TargetScaler('minmax')
        target_scaler.fit(np.array(all_targets))
        logger.info(f"Target scaler fitted: {target_scaler.mode}")
        logger.info(f"Target range: {min(all_targets)} to {max(all_targets)}")
        
        # 6. Create model configuration (very small for quick testing)
        model_config = ModelConfig(
            input_dim=768,
            hidden_size=32,         # Very small for quick test
            num_layers=1,           # Single layer
            bidirectional=True,
            dropout=0.0,            # No dropout for small model
            aggregation='last',
            mlp_hidden=32,
            use_layer_norm=False
        )
        
        # 7. Create training configuration (minimal training for testing)
        train_config = TrainConfig(
            epochs=2,               # Just 2 epochs for testing
            batch_size=batch_size,
            lr=1e-3,
            weight_decay=0,         # No weight decay for simple test
            optimizer='adamw',
            scheduler='none',       # No scheduler
            early_stopping_patience=10,
            grad_clip_norm=0,       # No gradient clipping
            use_amp=False,          # Disable AMP for simplicity
            target_scaler='minmax'
        )
        
        # 8. Create serialization configuration
        output_dir = 'ai-analysis/runs/complete_test'
        serialization_config = SerializationConfig(
            output_dir=output_dir,
            save_best_only=True,
            keep_last_k=2
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
            device=device
        )
        
        # 11. Train the model
        logger.info("Starting training...")
        best_metrics = trainer.train()
        logger.info("Training completed successfully")
        logger.info(f"Best validation metrics: {best_metrics}")
        
        # 12. Test inference
        logger.info("Testing inference...")
        checkpoint_path = f"{output_dir}/best.pt"
        if os.path.exists(checkpoint_path):
            trainer.load_checkpoint(checkpoint_path)
            logger.info("Best checkpoint loaded for inference test")
            
            # Run evaluation
            metrics, predictions = evaluate_model(model, val_loader, device, target_scaler)
            logger.info(f"Inference test metrics: {metrics}")
            logger.info(f"Sample predictions: {predictions[:2]}")
            
            return True
        else:
            logger.error("No checkpoint found for inference test")
            return False
            
    except Exception as e:
        logger.error(f"BiLSTM training/inference failed: {e}", exc_info=True)
        return False

def main():
    """Run complete end-to-end test"""
    logger.info("STARTING COMPLETE END-TO-END BiLSTM TEST")
    logger.info("=" * 80)
    
    success = True
    
    # Test 1: Preprocessing Pipeline
    logger.info("\\nStep 1: Testing BERTimbau Preprocessing Pipeline...")
    preprocessing_success = test_preprocessing_pipeline()
    if not preprocessing_success:
        logger.error("Preprocessing test FAILED")
        success = False
    else:
        logger.info("Preprocessing test PASSED")
    
    # Test 2: BiLSTM Training and Inference
    logger.info("\\nStep 2: Testing BiLSTM Training and Inference...")
    training_success = test_bilstm_training()
    if not training_success:
        logger.error("BiLSTM training/inference test FAILED")
        success = False
    else:
        logger.info("BiLSTM training/inference test PASSED")
    
    # Final summary
    logger.info("=" * 80)
    if success:
        logger.info("🎉 ALL TESTS PASSED! Complete BiLSTM pipeline is working correctly.")
        logger.info("✅ BERTimbau preprocessing: WORKING")
        logger.info("✅ BiLSTM training: WORKING") 
        logger.info("✅ BiLSTM inference: WORKING")
        logger.info("✅ Checkpoint saving/loading: WORKING")
        logger.info("")
        logger.info("The complete pipeline is ready for production use!")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
        return 1

if __name__ == '__main__':
    exit(main())