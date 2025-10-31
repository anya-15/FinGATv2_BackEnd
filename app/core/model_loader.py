"""
Model Loader - Loads FinGAT checkpoint with hot reload capability
"""

import torch
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.lightning_module import FinGATLightningModule
from app.config import settings


class ModelLoader:
    """
    Singleton model loader for FinGAT
    Supports hot-reloading after daily training
    """
    
    _instance = None
    _model: Optional[FinGATLightningModule] = None
    _metadata: Optional[Dict] = None
    _last_loaded: Optional[str] = None
    
    def __new__(cls):
        """Singleton pattern - only one instance"""
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, checkpoint_path: str = None) -> Tuple[FinGATLightningModule, Dict]:
        """
        Load FinGAT model from Lightning checkpoint
        
        Args:
            checkpoint_path: Path to .ckpt file
            
        Returns:
            (model, metadata) tuple
        """
        checkpoint_path = checkpoint_path or settings.MODEL_CHECKPOINT_PATH
        
        # Check if already loaded this checkpoint
        if self._model is not None and self._last_loaded == checkpoint_path:
            print("ℹ️ Model already loaded from this checkpoint")
            return self._model, self._metadata
        
        device = settings.DEVICE
        
        print(f"📦 Loading FinGAT model from {checkpoint_path}...")
        
        try:
            # Check if file exists
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Extract metadata
            if 'hyper_parameters' in checkpoint:
                hyper_params = checkpoint['hyper_parameters']
                self._metadata = hyper_params.get('metadata', {})
                config = hyper_params.get('config', {})
            else:
                self._metadata = {}
                config = {}
            
            # Load model from checkpoint
            self._model = FinGATLightningModule.load_from_checkpoint(
                checkpoint_path,
                map_location=device
            )
            
            # Set to evaluation mode
            self._model.eval()
            self._model.freeze()
            
            self._last_loaded = checkpoint_path
            
            print(f"✅ Model loaded successfully!")
            print(f"   Checkpoint: {checkpoint_path}")
            print(f"   Device: {device}")
            
            # Print model info if available
            if config:
                model_cfg = config.get('model', {})
                print(f"   Hidden dim: {model_cfg.get('hidden_dim', 'N/A')}")
                print(f"   Num heads: {model_cfg.get('num_heads', 'N/A')}")
                print(f"   Num layers: {model_cfg.get('num_layers', 'N/A')}")
            
            return self._model, self._metadata
            
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("💡 Train a model first: python scripts/train_model.py")
            raise
        
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def reload_model(self) -> Tuple[FinGATLightningModule, Dict]:
        """
        Reload model from checkpoint
        Used after daily training to load new model
        """
        print("\n🔄 Reloading model with latest checkpoint...")
        self._model = None
        self._last_loaded = None
        return self.load_model()
    
    def get_model(self) -> Tuple[FinGATLightningModule, Dict]:
        """
        Get loaded model
        
        Returns:
            (model, metadata) tuple
            
        Raises:
            RuntimeError if model not loaded
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() first or train a model."
            )
        return self._model, self._metadata
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model is not None


# Global singleton instance
model_loader = ModelLoader()
