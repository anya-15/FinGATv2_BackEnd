# FILE: train_with_hybrid_rl.py
"""
🚀 HYBRID RL: Feature Selection + Hyperparameter Tuning
Optimizes BOTH features and model hyperparameters
"""

import json
from datetime import datetime
import os
import sys
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# Ensure project root is on sys.path so local packages (data/, training/, etc.) can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.data_loader import FinancialDataset
from training.lightning_module import FinGATLightningModule, FinGATDataModule


class HybridOptimizationEnv(gym.Env):
    """
    RL Environment for HYBRID optimization
    State: Feature mask + hyperparameters
    Action: Modify features or hyperparameters
    Reward: Model accuracy improvement
    """
    
    def __init__(
        self, 
        config: dict,
        metadata: dict,
        data_module: FinGATDataModule,
        baseline_accuracy: float = 0.5652
    ):
        super().__init__()
        
        self.config = config
        self.metadata = metadata
        self.data_module = data_module
        self.baseline_accuracy = baseline_accuracy
        
        self.num_features = metadata['num_features']
        self.best_accuracy = baseline_accuracy
        self.best_features = None
        self.best_hparams = None
        
        # State space: [feature flags + hyperparameters]
        # Features: binary (use/don't use)
        # Hyperparams: hidden_dim (3 choices), dropout (3 choices), lr (3 choices)
        state_size = self.num_features + 9  # features + 3+3+3 hyperparams
        self.observation_space = spaces.MultiBinary(state_size)
        
        # Action space matches feature count plus hyperparameter options
        self.action_space = spaces.Discrete(self.num_features + 9)
        
        self.step_count = 0
        self.max_steps = 50  # Try 50 combinations
        
        # Start with current config
        self.selected_features = np.ones(self.num_features, dtype=int)
        self.current_hidden_dim = 128
        self.current_dropout = 0.3
        self.current_lr = 5e-4
        
        print(f"✅ Hybrid RL Environment initialized:")
        print(f"   Features: {self.num_features}")
        print(f"   Hyperparameters to optimize: 3 (hidden_dim, dropout, lr)")
        print(f"   Baseline accuracy: {self.baseline_accuracy:.4f}")
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.selected_features = np.ones(self.num_features, dtype=int)
        self.current_hidden_dim = 128
        self.current_dropout = 0.3
        self.current_lr = 5e-4
        self.step_count = 0
        
        return self._get_observation(), {}  # Gymnasium format: (obs, info)
    
    def _get_observation(self):
        """Get current state"""
        # Feature mask
        feature_state = self.selected_features.copy()
        
        # Hyperparameter state (one-hot encoded)
        hidden_dims = [64, 128, 256]
        hidden_dim_state = [1 if self.current_hidden_dim == dim else 0 for dim in hidden_dims]
        
        dropouts = [0.2, 0.3, 0.5]
        dropout_state = [1 if self.current_dropout == rate else 0 for rate in dropouts]
        
        learning_rates = [1e-4, 5e-4, 1e-3]
        lr_state = [1 if abs(self.current_lr - rate) < 1e-10 else 0 for rate in learning_rates]
        
        # Ensure one-hot property
        if sum(hidden_dim_state) == 0:
            hidden_dim_state[1] = 1  # Default to 128
        if sum(dropout_state) == 0:
            dropout_state[1] = 1  # Default to 0.3
        if sum(lr_state) == 0:
            lr_state[1] = 1  # Default to 5e-4
        
        state = np.concatenate([
            feature_state.astype(float),
            hidden_dim_state,
            dropout_state,
            lr_state
        ]).astype(np.float32)
        
        return state
    
    def step(self, action: int):
        """Execute one action"""
        
        self.step_count += 1
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PROCESS ACTION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Safety check for action bounds
        if action >= self.action_space.n:
            print(f"Warning: Action {action} out of bounds, ignoring")
            return self._get_observation(), -1.0, False, False, {
                'num_features': self.selected_features.sum(),
                'accuracy': 0.0,
                'hidden_dim': self.current_hidden_dim,
                'dropout': self.current_dropout,
                'lr': self.current_lr
            }
        if action < self.num_features:
            # Toggle a feature
            self.selected_features[action] = 1 - self.selected_features[action]
        elif action < self.num_features + 3:
            # Change hidden_dim
            hidden_idx = action - self.num_features
            self.current_hidden_dim = [64, 128, 256][hidden_idx]
        elif action < self.num_features + 6:
            # Change dropout
            dropout_idx = action - (self.num_features + 3)
            self.current_dropout = [0.2, 0.3, 0.5][dropout_idx]
        else:
            # Change learning_rate
            lr_idx = action - (self.num_features + 6)
            self.current_lr = [1e-4, 5e-4, 1e-3][lr_idx]
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # EVALUATE CONFIGURATION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        num_selected = self.selected_features.sum()

        # Penalty if too few features (less than 30% of total)
        min_features = int(self.num_features * 0.3)
        if num_selected < min_features:
            reward = -0.5
            done = False
            info = {
                'num_features': num_selected,
                'accuracy': 0.0,
                'hidden_dim': self.current_hidden_dim,
                'dropout': self.current_dropout,
                'lr': self.current_lr
            }
            return self._get_observation(), reward, False, False, info
        
        # Penalty if too many features (more than 90% of total)
        max_features = int(self.num_features * 0.9)
        if num_selected > max_features:
            reward = -0.2
            done = False
            info = {
                'num_features': num_selected,
                'accuracy': 0.0,
                'hidden_dim': self.current_hidden_dim,
                'dropout': self.current_dropout,
                'lr': self.current_lr
            }
            return self._get_observation(), reward, False, False, info
        
        # Train model with this configuration
        try:
            accuracy = self._evaluate_configuration()
        except Exception as e:
            print(f"Error: {e}")
            accuracy = self.baseline_accuracy
        
        # Calculate reward
        improvement = accuracy - self.baseline_accuracy
        
        # Bonus for efficiency
        feature_efficiency = 1.0 - (num_selected / self.num_features)
        
        reward = improvement + 0.05 * feature_efficiency
        
        # Update best
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_features = self.selected_features.copy()
            self.best_hparams = {
                'hidden_dim': self.current_hidden_dim,
                'dropout': self.current_dropout,
                'lr': self.current_lr,
                'num_features': num_selected
            }
            reward += 0.5
        
        done = (self.step_count >= self.max_steps) or (accuracy > 0.70)
        
        info = {
            'num_features': num_selected,
            'accuracy': accuracy,
            'best_accuracy': self.best_accuracy,
            'hidden_dim': self.current_hidden_dim,
            'dropout': self.current_dropout,
            'lr': self.current_lr
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _evaluate_configuration(self) -> float:
        """Train model with current configuration"""
        
        selected_indices = np.where(self.selected_features == 1)[0]
        
        print(f"\n🔍 Testing: {len(selected_indices)} features | "
              f"hidden={self.current_hidden_dim} | "
              f"dropout={self.current_dropout} | "
              f"lr={self.current_lr:.1e}")
        
        # Mask data
        train_data = self.data_module.train_data.clone()
        val_data = self.data_module.val_data.clone()
        
        train_data.x = train_data.x[:, selected_indices]
        val_data.x = val_data.x[:, selected_indices]
        
        # Update config
        config = self.config.copy()
        config['model']['input_dim'] = len(selected_indices)
        config['model']['hidden_dim'] = self.current_hidden_dim
        config['model']['dropout'] = self.current_dropout
        config['training']['learning_rate'] = self.current_lr
        
        # Quicker training (2 epochs to save time)
        model = FinGATLightningModule(config, self.metadata)
        
        data_module = FinGATDataModule(
            config=config,
            train_data=train_data,
            val_data=val_data,
            test_data=self.data_module.test_data,
            metadata=self.metadata
        )
        
        trainer = pl.Trainer(
            max_epochs=3,
            accelerator='auto',
            devices='auto',
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False
        )
        
        trainer.fit(model, data_module)
        
        val_results = trainer.validate(model, data_module)
        accuracy = val_results[0]['val_accuracy']
        
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        
        return accuracy


def train_with_hybrid_rl():
    """Main hybrid training function"""
    
    print("="*80)
    print("🚀 HYBRID RL: FEATURE SELECTION + HYPERPARAMETER TUNING")
    print("="*80)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 1: PREPARE DATA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n📊 Step 1: Preparing dataset...")
    
    dataset = FinancialDataset(
        csv_folder_path="indian_data",
        max_stocks=550
    )
    data, metadata = dataset.prepare_dataset()
    
    train_data, val_data, test_data = dataset.create_temporal_splits(
        data, metadata, train_ratio=0.7, val_ratio=0.15
    )
    
    print(f"✅ Dataset ready: {metadata['num_features']} features, 73 temporal")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 2: BASELINE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n📊 Step 2: Baseline (current model)...")
    
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
            'max_epochs': 5,
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
        max_epochs=5,
        accelerator='auto',
        devices='auto',
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False
    )
    
    trainer.fit(baseline_model, data_module)
    val_results = trainer.validate(baseline_model, data_module)
    baseline_accuracy = val_results[0]['val_accuracy']
    
    print(f"✅ Baseline accuracy: {baseline_accuracy:.4f}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 3: HYBRID RL OPTIMIZATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n🤖 Step 3: Running HYBRID RL optimization...")
    print("   Optimizing: Features + Hyperparameters")
    
    env = HybridOptimizationEnv(
        config=config,
        metadata=metadata,
        data_module=data_module,
        baseline_accuracy=baseline_accuracy
    )
    
    agent = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        n_steps=128,  # Reduced from 256
        batch_size=32,
        n_epochs=2,   # Reduced from 3
        ent_coef=0.01,
        verbose=1,
        device='auto'
    )
    
    agent.learn(total_timesteps=1000)  # Reduced from 2000
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 4: FINAL TRAINING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n📊 Step 4: Final training with optimized config...")
    
    # Force save artifacts even if no improvement
    if env.best_features is None:
        print("\n⚠️ No improvement found - using latest explored configuration")
        env.best_features = env.selected_features.copy()
        env.best_hparams = {
            'hidden_dim': env.current_hidden_dim,
            'dropout': env.current_dropout,
            'lr': env.current_lr,
            'num_features': int(env.selected_features.sum())
        }
    
    # Proceed with final training using best or latest config
    selected_indices = np.where(env.best_features == 1)[0]
    best_hparams = env.best_hparams
        
    # Mask data
    train_data_opt = train_data.clone()
    val_data_opt = val_data.clone()
    test_data_opt = test_data.clone()
    
    train_data_opt.x = train_data_opt.x[:, selected_indices]
    val_data_opt.x = val_data_opt.x[:, selected_indices]
    test_data_opt.x = test_data_opt.x[:, selected_indices]
    
    # Update config
    config_opt = config.copy()
    config_opt['model']['input_dim'] = len(selected_indices)
    config_opt['model']['hidden_dim'] = best_hparams['hidden_dim']
    config_opt['model']['dropout'] = best_hparams['dropout']
    config_opt['training']['learning_rate'] = best_hparams['lr']
    config_opt['training']['max_epochs'] = 30  # Reduced from 50
    
    final_model = FinGATLightningModule(config_opt, metadata)
    
    data_module_opt = FinGATDataModule(
        config=config_opt,
        train_data=train_data_opt,
        val_data=val_data_opt,
        test_data=test_data_opt,
        metadata=metadata
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename='fingat-hybrid-{epoch:02d}-{val_mrr:.4f}',
        monitor='val_mrr',
        mode='max',
        save_top_k=3
    )
    
    early_stop = EarlyStopping(
        monitor='val_mrr',
        patience=15,
        mode='max'
    )
    
    trainer_final = pl.Trainer(
        max_epochs=50,
        accelerator='auto',
        devices='auto',
        callbacks=[checkpoint_callback, early_stop],
        logger=CSVLogger("logs", name="fingat-hybrid"),
        enable_progress_bar=True,
        gradient_clip_val=1.0
    )
    
    trainer_final.fit(final_model, data_module_opt)
    trainer_final.test(final_model, data_module_opt)
    
    val_results_final = trainer_final.validate(final_model, data_module_opt)
    final_accuracy = val_results_final[0]['val_accuracy']

    # Always save artifacts
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_root = os.path.join("rl_models", "hybrid", run_id)
    os.makedirs(save_root, exist_ok=True)

    # 1) Save feature mask
    mask_path = os.path.join(save_root, "best_features.npy")
    np.save(mask_path, env.best_features.astype(np.int8))
    print(f"✓ Saved feature mask: {mask_path}")

    # 2) Save hyperparameters 
    hparams = {
        "hidden_dim": env.best_hparams["hidden_dim"],
        "dropout": env.best_hparams["dropout"],
        "learning_rate": env.best_hparams["lr"],
        "num_features": int(env.best_hparams["num_features"]),
        "val_accuracy": float(final_accuracy)
    }
    hparams_path = os.path.join(save_root, "best_hparams.json")
    with open(hparams_path, "w") as f:
        json.dump(hparams, f, indent=2)
    print(f"✓ Saved hyperparameters: {hparams_path}")

    # 3) Resolve best checkpoint path
    best_ckpt = None
    for cb in trainer_final.callbacks:
        if isinstance(cb, ModelCheckpoint):
            best_ckpt = cb.best_model_path
            break
    if best_ckpt is None or not os.path.exists(best_ckpt):
        # Fallback: pick the newest checkpoint
        import glob
        cands = glob.glob(os.path.join("checkpoints", "fingat-hybrid-*.ckpt"))
        cands.sort(key=os.path.getmtime, reverse=True)
        best_ckpt = cands[0] if cands else ""

    ckpt_txt = os.path.join(save_root, "best_checkpoint_path.txt")
    with open(ckpt_txt, "w") as f:
        f.write(best_ckpt)
    print(f"✓ Saved checkpoint reference: {ckpt_txt}")

    # 4) Write run manifest
    # Convert paths to relative and use forward slashes for cross-platform compatibility
    def normalize_path(path):
        """Convert to relative path with forward slashes"""
        # Get relative path from project root
        try:
            rel_path = os.path.relpath(path)
        except ValueError:
            # If on different drives (Windows), use the path as-is
            rel_path = path
        # Replace backslashes with forward slashes
        return rel_path.replace("\\", "/")
    
    manifest = {
        "run_id": run_id,
        "features_path": normalize_path(mask_path),
        "hparams_path": normalize_path(hparams_path),
        "checkpoint_path": normalize_path(best_ckpt),
        "notes": "Hybrid RL: features + hparams; final 50-epoch training",
    }
    manifest_path = os.path.join(save_root, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✓ Saved run manifest: {manifest_path}")

    # 5) Update global latest manifest for predict_now.py
    latest_dir = os.path.join("rl_models", "selected_runs")
    os.makedirs(latest_dir, exist_ok=True)
    latest_manifest = os.path.join(latest_dir, "latest_manifest.json")
    with open(latest_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✓ Updated latest manifest: {latest_manifest}")
    print(f"✓ Selected features: {int(env.best_features.sum())}")
    print(f"✓ Best checkpoint: {os.path.basename(best_ckpt)}")

    # Set final accuracy value
    final_accuracy = baseline_accuracy if not trainer_final else trainer_final.callback_metrics.get("val/accuracy", baseline_accuracy)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RESULTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n" + "="*80)
    print("📊 HYBRID OPTIMIZATION RESULTS")
    print("="*80)
    
    print(f"\nBaseline (73 features, default hyperparameters):")
    print(f"   Accuracy: {baseline_accuracy:.4f}")
    
    if env.best_hparams:
        print(f"\nOptimized Configuration:")
        print(f"   Features: {env.best_hparams['num_features']}")
        print(f"   Hidden Dim: {env.best_hparams['hidden_dim']}")
        print(f"   Dropout: {env.best_hparams['dropout']}")
        print(f"   Learning Rate: {env.best_hparams['lr']:.1e}")
        print(f"   Accuracy: {final_accuracy:.4f}")
        
        improvement = (final_accuracy - baseline_accuracy) / baseline_accuracy * 100
        print(f"\n🎉 Improvement: {improvement:+.2f}%")
    
    print("\n✅ Hybrid training complete!")


if __name__ == "__main__":
    train_with_hybrid_rl()