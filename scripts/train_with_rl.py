
# FILE 3: train_with_rl.py
"""
Complete Training Script with RL Feature Selection
"""

import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from data.data_loader import FinancialDataset
from training.lightning_module import FinGATLightningModule, FinGATDataModule
from rl.feature_env import FeatureSelectionEnv
from rl.feature_selector import FeatureSelector


def train_with_rl():
    """
    ✅ Complete workflow:
    1. Load data with multi-scale features (73 features)
    2. Train RL agent to select best features (~40-50)
    3. Train final FinGAT with selected features
    4. Compare accuracy
    """
    
    print("="*80)
    print("🚀 FINGAT + RL FEATURE SELECTION")
    print("="*80)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 1: PREPARE DATA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n📊 Step 1: Preparing dataset...")
    
    dataset = FinancialDataset(
        csv_folder_path="indian_data",
        max_stocks=550
    )
    data, metadata = dataset.prepare_dataset()
    
    train_data, val_data, test_data = dataset.create_temporal_splits(
        data, metadata, train_ratio=0.7, val_ratio=0.15
    )
    
    print(f"✅ Dataset prepared:")
    print(f"   Train: {train_data.num_nodes} stocks")
    print(f"   Val: {val_data.num_nodes} stocks")
    print(f"   Test: {test_data.num_nodes} stocks")
    print(f"   Features: {metadata['num_features']}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 2: BASELINE TRAINING (with all 73 features)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n📊 Step 2: Training baseline model (73 features)...")
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("rl_models", exist_ok=True)
    
    config = {
        'model': {
            'input_dim': metadata['num_features'],
            'hidden_dim': 128,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.3,
            'output_dim': 3,
            'use_residual': True,
            'use_temporal': False
        },
        'training': {
            'learning_rate': 5e-4,
            'weight_decay': 1e-4,
            'max_epochs': 10,  # Quick baseline
            'patience': 30
        }
    }
    
    baseline_model = FinGATLightningModule(config, metadata)
    data_module = FinGATDataModule(
        config=config,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        metadata=metadata
    )
    
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='auto',
        devices='auto',
        enable_progress_bar=True,
        logger=CSVLogger("logs", name="baseline"),
        enable_checkpointing=False
    )
    
    trainer.fit(baseline_model, data_module)
    baseline_val_results = trainer.validate(baseline_model, data_module)
    baseline_accuracy = baseline_val_results[0]['val_accuracy']
    
    print(f"\n✅ Baseline model trained:")
    print(f"   Features: 73")
    print(f"   Validation accuracy: {baseline_accuracy:.4f}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 3: RL FEATURE SELECTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n🤖 Step 3: Running RL feature selector...")
    print("   This will try different feature combinations...")
    
    env = FeatureSelectionEnv(
        config=config,
        metadata=metadata,
        data_module=data_module,
        baseline_accuracy=baseline_accuracy,
        min_features=30,
        max_features=60
    )
    
    selector = FeatureSelector(env)
    selector.train(total_timesteps=5000)
    
    # Get best features
    best_features = selector.get_best_features()
    selector.save_best_features()
    
    selected_indices = np.where(best_features == 1)[0]
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 4: FINAL TRAINING (with RL-selected features)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n📊 Step 4: Training final model with RL-selected features...")
    
    # Mask data to selected features
    train_data_selected = train_data.clone()
    val_data_selected = val_data.clone()
    test_data_selected = test_data.clone()
    
    train_data_selected.x = train_data_selected.x[:, selected_indices]
    val_data_selected.x = val_data_selected.x[:, selected_indices]
    test_data_selected.x = test_data_selected.x[:, selected_indices]
    
    # Update config
    config_final = config.copy()
    config_final['model']['input_dim'] = len(selected_indices)
    config_final['training']['max_epochs'] = 50
    
    final_model = FinGATLightningModule(config_final, metadata)
    
    data_module_final = FinGATDataModule(
        config=config_final,
        train_data=train_data_selected,
        val_data=val_data_selected,
        test_data=test_data_selected,
        metadata=metadata
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename='fingat-rl-{epoch:02d}-{val_mrr:.4f}',
        monitor='val_mrr',
        mode='max',
        save_top_k=3
    )
    
    early_stop = EarlyStopping(
        monitor='val_mrr',
        patience=20,
        mode='max'
    )
    
    trainer_final = pl.Trainer(
        max_epochs=config_final['training']['max_epochs'],
        accelerator='auto',
        devices='auto',
        callbacks=[checkpoint_callback, early_stop],
        logger=CSVLogger("logs", name="fingat-rl"),
        enable_progress_bar=True,
        gradient_clip_val=1.0
    )
    
    trainer_final.fit(final_model, data_module_final)
    trainer_final.test(final_model, data_module_final)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 5: COMPARISON REPORT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    val_results_final = trainer_final.validate(final_model, data_module_final)
    final_accuracy = val_results_final[0]['val_accuracy']
    
    print("\n" + "="*80)
    print("📊 COMPARISON: BASELINE vs RL-OPTIMIZED")
    print("="*80)
    
    print(f"\nBaseline Model (73 features):")
    print(f"   Accuracy: {baseline_accuracy:.4f}")
    print(f"   Features: 73")
    
    print(f"\nRL-Optimized Model ({len(selected_indices)} features):")
    print(f"   Accuracy: {final_accuracy:.4f}")
    print(f"   Features: {len(selected_indices)}")
    
    improvement = (final_accuracy - baseline_accuracy) / baseline_accuracy * 100
    print(f"\n🎉 Improvement: {improvement:+.2f}%")
    print(f"   New accuracy: {final_accuracy:.4f} (was {baseline_accuracy:.4f})")
    
    # Save report
    report = f"""
RL FEATURE SELECTION REPORT
============================

Baseline (73 features):
  Accuracy: {baseline_accuracy:.4f}

RL-Optimized ({len(selected_indices)} features):
  Accuracy: {final_accuracy:.4f}
  Selected features: {selected_indices.tolist()}

Improvement: {improvement:+.2f}%

Best features saved to: rl_models/best_features.npy
Final model saved to: {checkpoint_callback.best_model_path}
"""
    
    with open("rl_models/report.txt", "w") as f:
        f.write(report)
    
    print(f"\n✅ Report saved to: rl_models/report.txt")
    print("="*80)


if __name__ == "__main__":
    train_with_rl()