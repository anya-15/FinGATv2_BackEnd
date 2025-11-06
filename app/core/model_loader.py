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
        If no path is provided, automatically loads the latest RL-trained checkpoint from manifest.
        """
        import json
        
        # Get project root
        project_root = Path(__file__).parent.parent.parent
        manifest_path = project_root / 'rl_models' / 'selected_runs' / 'latest_manifest.json'
        
        # If no explicit path, use latest manifest
        if checkpoint_path is None:
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    checkpoint_path = manifest.get('checkpoint_path')
                    # Make path absolute if it's relative
                    if checkpoint_path and not Path(checkpoint_path).is_absolute():
                        checkpoint_path = str(project_root / checkpoint_path)
                    print(f"ℹ️ Auto-selected checkpoint from manifest: {checkpoint_path}")
                except Exception as e:
                    print(f"⚠️ Error reading manifest: {e}")
                    checkpoint_path = None
            
            if checkpoint_path is None:
                checkpoint_path = getattr(settings, 'MODEL_CHECKPOINT_PATH', None)
                if checkpoint_path and not Path(checkpoint_path).is_absolute():
                    checkpoint_path = str(project_root / checkpoint_path)
                print(f"⚠️ Manifest not found, falling back to config: {checkpoint_path}")

        # Check if already loaded this checkpoint
        if self._model is not None and self._last_loaded == checkpoint_path:
            print("ℹ️ Model already loaded from this checkpoint")
            return self._model, self._metadata

        # Get device configuration
        device = getattr(settings, 'DEVICE', 'cpu')
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️ CUDA requested but not available, falling back to CPU")
            device = 'cpu'

        print(f"📦 Loading FinGAT model from {checkpoint_path}...")

        try:
            # Check if file exists
            if not checkpoint_path:
                raise FileNotFoundError(
                    "No checkpoint path provided. Set MODEL_CHECKPOINT_PATH in .env or train a model first."
                )
            
            checkpoint_file = Path(checkpoint_path)
            if not checkpoint_file.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}\n"
                    f"Please ensure the file exists or train a model first."
                )

            # Load checkpoint
            print(f"   Loading checkpoint file...")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            # Extract metadata
            if 'hyper_parameters' in checkpoint:
                hyper_params = checkpoint['hyper_parameters']
                self._metadata = hyper_params.get('metadata', {})
                config = hyper_params.get('config', {})
            else:
                print("⚠️ No hyper_parameters found in checkpoint, using empty metadata")
                self._metadata = {}
                config = {}

            # Load model from checkpoint
            print(f"   Initializing model from checkpoint...")
            self._model = FinGATLightningModule.load_from_checkpoint(
                checkpoint_path,
                map_location=device
            )

            # Move model to device
            self._model = self._model.to(device)
            
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
        
        except RuntimeError as e:
            if "PytorchStreamReader" in str(e) or "not a zip file" in str(e):
                print(f"❌ Checkpoint file is corrupted or invalid: {checkpoint_path}")
                print("💡 Please retrain the model or use a different checkpoint")
            else:
                print(f"❌ Runtime error loading model: {e}")
            raise

        except Exception as e:
            print(f"❌ Unexpected error loading model: {e}")
            import traceback
            print(traceback.format_exc())
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
