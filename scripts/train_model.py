"""
Train FinGAT Model - Production Script
Trains model and saves to checkpoints/production_model.ckpt
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime

from training.lightning_module import FinGATLightningModule, FinGATDataModule
from data.data_loader import FinancialDataset
from app.config import model_config, settings


def train_fingat_model():
    """
    Train FinGAT model with Indian stock data
    Uses your existing FinGATLightningModule
    """
    
    print("=" * 60)
    print("🏋️ FinGAT Model Training - Indian Stocks")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Ensure directories exist
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Training config
    training_cfg = model_config.get('training', {})
    
    print("\n📊 Configuration:")
    print(f"   Data path: {settings.DATA_PATH}")
    print(f"   Device: {settings.DEVICE}")
    print(f"   Max epochs: {training_cfg.get('max_epochs', 50)}")
    print(f"   Learning rate: {training_cfg.get('learning_rate', 0.001)}")
    print(f"   Batch size: {training_cfg.get('batch_size', 16)}")
    
    # ============================================
    # STEP 1: Prepare Dataset
    # ============================================
    print("\n🔧 Preparing dataset...")
    dataset = FinancialDataset(
        csv_folder_path=settings.DATA_PATH,
        max_stocks=model_config.get('data', {}).get('max_stocks', 550)
    )
    
    # Prepare full dataset
    data, metadata = dataset.prepare_dataset()
    
    print(f"✅ Dataset prepared:")
    print(f"   Total stocks: {data.num_nodes}")
    print(f"   Features per stock: {metadata['num_features']}")
    print(f"   Graph edges: {data.edge_index.size(1)}")
    
    # ============================================
    # STEP 2: Create Train/Val/Test Splits
    # ============================================
    print("\n🔧 Creating train/val/test splits...")
    train_data, val_data, test_data = dataset.create_temporal_splits(
        data, 
        metadata,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    print(f"✅ Splits created:")
    print(f"   Train stocks: {train_data.num_nodes}")
    print(f"   Val stocks: {val_data.num_nodes}")
    print(f"   Test stocks: {test_data.num_nodes}")
    
    # ============================================
    # STEP 3: Initialize Model with METADATA
    # ============================================
    print("\n🔧 Initializing FinGAT Lightning Module...")
    model = FinGATLightningModule(
        config=model_config,
        metadata=metadata  # ← FIXED: Now passing metadata
    )
    
    # ============================================
    # STEP 4: Initialize Data Module
    # ============================================
    datamodule = FinGATDataModule(
        config=model_config,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        metadata=metadata
    )
    
    # ============================================
    # STEP 5: Setup Callbacks
    # ============================================
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="production_model",
        save_top_k=1,
        monitor='val_mrr',
        mode='max',
        save_last=False,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_mrr',
        patience=training_cfg.get('patience', 15),
        mode='max',
        verbose=True
    )
    
    # ============================================
    # STEP 6: Setup Logger
    # ============================================
    logger = TensorBoardLogger(
        save_dir="logs",
        name="fingat_training",
        version=datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    
    # ============================================
    # STEP 7: Create Trainer
    # ============================================
    trainer = pl.Trainer(
        max_epochs=training_cfg.get('max_epochs', 50),
        accelerator=training_cfg.get('accelerator', 'cpu'),
        devices=training_cfg.get('devices', 'auto'),
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        gradient_clip_val=training_cfg.get('gradient_clip_val', 1.0),
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True
    )
    
    print("\n🚀 Starting training...")
    print("=" * 60)
    
    try:
        # Train model
        trainer.fit(model, datamodule)
        
        # Test model
        print("\n📊 Running final evaluation...")
        trainer.test(model, datamodule, ckpt_path='best')
        
        print("\n" + "=" * 60)
        print("✅ Training completed successfully!")
        print(f"   Best model saved to: checkpoints/production_model.ckpt")
        print(f"   Logs saved to: logs/fingat_training/")
        print("=" * 60)
        
        return True
    
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Training failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = train_fingat_model()
    sys.exit(0 if success else 1)
