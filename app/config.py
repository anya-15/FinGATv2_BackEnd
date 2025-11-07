"""
Backend API Configuration
Loads settings from .env file and config.yaml
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import os


class Settings(BaseSettings):
    """API Settings from .env file"""
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/fingat_db"
    
    # Paths
    CONFIG_PATH: str = "config/config.yaml"
    MODEL_CHECKPOINT_PATH: str = "checkpoints/production_model.ckpt"
    DATA_PATH: str = "indian_data"
    
    # Runtime
    DEVICE: str = "cpu"
    PYTHONPATH: str = "."
    
    # API
    PROJECT_NAME: str = "FinGAT Indian Stock Prediction API"
    API_V1_PREFIX: str = "/api/v1"
    API_PORT: int = 8000
    
    # Training Schedule
    TRAINING_HOUR: int = 18
    TRAINING_MINUTE: int = 30
    TRAINING_TIMEZONE: str = "Asia/Kolkata"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    ENABLE_WANDB: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


@lru_cache()
def load_model_config() -> Dict[str, Any]:
    """
    Load model configuration from config.yaml
    This is used for training and model initialization
    """
    settings = get_settings()
    config_path = Path(settings.CONFIG_PATH)
    
    if not config_path.exists():
        print(f"[WARNING] Config file not found at {config_path}, using defaults")
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"[OK] Loaded config from {config_path}")
        return config
    
    except Exception as e:
        print(f"[WARNING] Error loading config: {e}, using defaults")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Fallback configuration if config.yaml not found"""
    return {
        'model': {
            'name': 'FinGAT_Indian_2025',
            'input_dim': 36,
            'hidden_dim': 128,
            'output_dim': 1,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.3,
            'use_residual': True,
            'use_temporal': True
        },
        'training': {
            'optimizer': 'AdamW',
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'max_epochs': 50,
            'patience': 15,
            'gradient_clip_val': 1.0,
            'accelerator': 'cpu'
        },
        'data': {
            'market': 'INDIA',
            'csv_folder_path': 'indian_data',
            'max_stocks': 550,
            'lookback_window': 20,
            'prediction_horizon': 5,
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15
        },
        'paths': {
            'data_dir': 'indian_data/',
            'checkpoints_dir': 'checkpoints/',
            'results_dir': 'results/',
            'logs_dir': 'logs/'
        }
    }


# Global instances
settings = get_settings()
model_config = load_model_config()
