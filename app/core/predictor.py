"""
Predictor - Top-K Stock Prediction using FinGAT, sector-aware version
"""

import torch
from torch_geometric.data import Data
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from pathlib import Path
import pandas as pd
import sys
import traceback
import logging
import os
from contextlib import redirect_stdout, redirect_stderr

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.core.model_loader import model_loader
from app.config import settings
from data.data_loader import FinancialDataset
from app.db.models import Stock, Prediction

# Setup logger
logger = logging.getLogger(__name__)

class TopKPredictor:
    def __init__(self):
        self.model = None
        self.metadata = None
        self.data_loader = None
        self.device = torch.device(settings.DEVICE)
        self._initialized = False
        self.feature_mask = None  # Cache for RL feature mask
        self._feature_mask_loaded = False
        self._cached_data = None  # Cache prepared dataset
        self._cached_metadata = None  # Cache dataset metadata
        self._cache_timestamp = None  # Track when cache was created

    def _ensure_initialized(self):
        if self._initialized:
            return
        try:
            # Suppress stdout/stderr to avoid Windows pipe errors
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    self.model, self.metadata = model_loader.get_model()
        except RuntimeError:
            return
        
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                self.data_loader = FinancialDataset(
                    csv_folder_path=settings.DATA_PATH,
                    max_stocks=550
                )
        self._initialized = True
    
    def _load_feature_mask(self):
        """
        Load RL feature mask from latest manifest.
        This is called automatically on every prediction to ensure we always use
        the correct features from the most recent training run.
        """
        if self._feature_mask_loaded:
            return  # Already loaded in this session
        
        try:
            import numpy as np
            from pathlib import Path
            import json
            
            project_root = Path(__file__).parent.parent.parent
            manifest_path = project_root / 'rl_models' / 'selected_runs' / 'latest_manifest.json'
            
            if not manifest_path.exists():
                logger.info("No RL manifest found - using all features")
                self._feature_mask_loaded = True
                return
            
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Try both 'features_path' and 'feature_mask_path' keys
            feature_mask_path = manifest.get('features_path') or manifest.get('feature_mask_path')
            
            if not feature_mask_path:
                logger.info("No feature mask path in manifest - using all features")
                self._feature_mask_loaded = True
                return
            
            # Handle relative paths
            if not Path(feature_mask_path).is_absolute():
                feature_mask_path = project_root / feature_mask_path
            
            if not Path(feature_mask_path).exists():
                logger.warning(f"Feature mask file not found: {feature_mask_path}")
                self._feature_mask_loaded = True
                return
            
            # Load and cache the feature mask
            feature_mask = np.load(feature_mask_path)
            self.feature_mask = torch.from_numpy(feature_mask).bool().to(self.device)
            self._feature_mask_loaded = True
            
            num_selected = int(feature_mask.sum())
            logger.info(f"[OK] Loaded RL feature mask: {num_selected}/{len(feature_mask)} features selected")
            logger.info(f"   Mask: {feature_mask.tolist()}")
            
        except Exception as e:
            logger.error(f"Error loading feature mask: {e}")
            self._feature_mask_loaded = True  # Don't keep trying if it fails
    
    def reload_feature_mask(self):
        """
        Force reload the feature mask from the latest manifest.
        Call this after training a new model to immediately use the new features.
        """
        self._feature_mask_loaded = False
        self.feature_mask = None
        self._load_feature_mask()
        logger.info("[*] Feature mask reloaded from latest training run")

    @torch.no_grad()
    def predict_top_k(
        self, db: Optional[Session] = None, k: int = 10, sector: Optional[str] = None
    ) -> List[Dict]:
        self._ensure_initialized()
        if not self._initialized:
            try:
                logger.info("[*] Initializing predictor...")
                self.model, self.metadata = model_loader.get_model()
                logger.info(f"[OK] Model loaded: {type(self.model)}")
                self.data_loader = FinancialDataset(
                    csv_folder_path=settings.DATA_PATH,
                    max_stocks=550
                )
                logger.info(f"[OK] Data loader initialized for path: {settings.DATA_PATH}")
                self._initialized = True
            except Exception as e:
                logger.error(f"[ERROR] Cannot initialize predictor: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return []
        try:
            # Use cached dataset if available (cache for 5 minutes)
            import time
            current_time = time.time()
            cache_duration = 5 * 60  # 5 minutes
            
            if (self._cached_data is not None and 
                self._cache_timestamp is not None and 
                (current_time - self._cache_timestamp) < cache_duration):
                # Use cached data
                data = self._cached_data
                metadata = self._cached_metadata
                logger.info(f"[CACHE] Using cached dataset ({data.x.shape[0]} stocks)")
            else:
                # Load fresh data
                logger.info("[*] Preparing dataset...")
                with open(os.devnull, 'w') as devnull:
                    with redirect_stdout(devnull), redirect_stderr(devnull):
                        data, metadata = self.data_loader.prepare_dataset()
                
                # Cache the data
                self._cached_data = data
                self._cached_metadata = metadata
                self._cache_timestamp = current_time
                logger.info(f"[OK] Dataset prepared and cached: {data.x.shape[0]} stocks, {data.x.shape[1]} features")
        except Exception as e:
            logger.error(f"[ERROR] Error preparing dataset: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        
        # Load and apply RL feature mask (auto-syncs with latest training)
        self._load_feature_mask()
        if self.feature_mask is not None:
            x = x[:, self.feature_mask]

        # Check for correct graph attributes
        missing = []
        for field in ["stock_to_sector", "sector_edge_index"]:
            if not hasattr(data, field):
                missing.append(field)
        if missing:
            raise AttributeError(
                f"Missing sector-aware graph fields: {missing}. "
                "You MUST ensure your data loader populates these for both training and prediction."
            )

        stock_to_sector = data.stock_to_sector.to(self.device)
        sector_edge_index = data.sector_edge_index.to(self.device)

        returns_pred, movement_pred, ranking_scores, embeddings = self.model(
            x, edge_index, stock_to_sector, sector_edge_index
        )

        ranking_scores = ranking_scores.squeeze()
        if sector and sector.lower() != "all":
            sector_mask = self._get_sector_mask(data, metadata, sector, db)
            ranking_scores = ranking_scores * sector_mask
        k_actual = min(k, len(ranking_scores))
        top_k_values, top_k_indices = torch.topk(ranking_scores, k=k_actual)
        movement_probs = torch.softmax(movement_pred, dim=1)
        movement_predictions = torch.argmax(movement_pred, dim=1)
        results = []
        ticker_to_idx = metadata.get('ticker_to_idx', {})
        idx_to_ticker = {v: k for k, v in ticker_to_idx.items()}
        for rank, idx in enumerate(top_k_indices, 1):
            idx = idx.item()
            ticker = idx_to_ticker.get(idx, f"STOCK_{idx}")
            stock = None
            if db:
                stock = db.query(Stock).filter(Stock.ticker == ticker).first()
            company_name = stock.company_name if (stock and stock.company_name) else ticker
            predicted_return = returns_pred[idx].item()
            movement_class = movement_predictions[idx].item()
            movement_confidence = movement_probs[idx][movement_class].item()
            ranking_score = top_k_values[rank-1].item()
            predicted_movement = "up" if movement_class == 1 else "down"
            movement_percentage = abs(predicted_return * 100)
            price_date = "2025-10-29"
            latest_price = 0.0
            csv_path = Path(settings.DATA_PATH) / f"{ticker}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if 'Date' in df.columns:
                        latest_date_str = str(df['Date'].iloc[-1])
                        price_date = latest_date_str.split()[0]
                    if 'Close' in df.columns:
                        latest_price = float(df['Close'].iloc[-1])
                except Exception as e:
                    logger.warning(f"Error reading {ticker}.csv: {e}")
                    latest_price = float(stock.current_price) if stock and stock.current_price else 0.0
            else:
                latest_price = float(stock.current_price) if stock and stock.current_price else 0.0
            result_dict = {
                "rank": rank,
                "ticker": ticker,
                "company_name": company_name,
                "current_price": round(latest_price, 2),
                "price_date": price_date,
                "predicted_movement": predicted_movement,
                "movement_percentage": round(movement_percentage, 2),
                "confidence_score": round(movement_confidence, 3),
                "ranking_score": round(ranking_score, 3),
                "predicted_return": round(predicted_return, 4),
                "sector": stock.sector if stock else "Unknown"
            }
            results.append(result_dict)
            
            # Save prediction to database
            if db:
                self._save_prediction_to_db(db, result_dict)
        
        return results
    
    def _save_prediction_to_db(self, db: Session, prediction_data: Dict):
        """Save a single prediction to the database"""
        try:
            # Get model checkpoint info
            checkpoint_path = None
            model_version = None
            try:
                import json
                manifest_path = Path(__file__).parent.parent.parent / 'rl_models' / 'selected_runs' / 'latest_manifest.json'
                if manifest_path.exists():
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    checkpoint_path = manifest.get('checkpoint_path', '')
                    if checkpoint_path:
                        checkpoint_path = Path(checkpoint_path).name  # Just the filename
                    model_version = manifest.get('run_id', 'unknown')
            except:
                pass
            
            prediction = Prediction(
                ticker=prediction_data['ticker'],
                predicted_return=prediction_data['predicted_return'],
                predicted_movement=prediction_data['predicted_movement'],
                movement_confidence=prediction_data['confidence_score'],
                ranking_score=prediction_data['ranking_score'],
                rank=prediction_data['rank'],
                model_checkpoint=checkpoint_path,
                model_version=model_version,
                sector=prediction_data['sector']
            )
            
            db.add(prediction)
            db.commit()
            logger.debug(f"Saved prediction for {prediction_data['ticker']} to database")
        except Exception as e:
            logger.error(f"Error saving prediction to database: {e}")
            db.rollback()

    def _get_sector_mask(self, data: Data, metadata: Dict, sector: str, db: Optional[Session]) -> torch.Tensor:
        num_stocks = data.num_nodes
        mask = torch.zeros(num_stocks, device=self.device)
        ticker_to_idx = metadata.get('ticker_to_idx', {})
        if not db:
            return mask
        sector_stocks = db.query(Stock).filter(Stock.sector == sector).all()
        for stock in sector_stocks:
            if stock.ticker in ticker_to_idx:
                idx = ticker_to_idx[stock.ticker]
                mask[idx] = 1.0
        return mask

_predictor_instance = None

def get_predictor() -> TopKPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = TopKPredictor()
    return _predictor_instance

def run_predict_now(top_k_values=[5, 10, 20]) -> Dict:
    try:
        # Monkey-patch print to completely suppress output and avoid Windows pipe errors
        import builtins
        original_print = builtins.print
        
        def silent_print(*args, **kwargs):
            """Silent print that does nothing"""
            pass
        
        try:
            # Replace print globally
            builtins.print = silent_print
            
            # Also redirect stdout/stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            
            predictor = get_predictor()
            up_predictions = predictor.predict_top_k(db=None, k=1000, sector=None)
            
        finally:
            # Restore everything
            builtins.print = original_print
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        result = {
            "top_k_results": {},
            "full_up_predictions": up_predictions,
        }
        for k in top_k_values:
            result["top_k_results"][f"top_{k}"] = up_predictions[:k]
        result["batch_count"] = len(up_predictions)
        return result
    except OSError as e:
        if "pipe" in str(e).lower():
            return {
                "error": "Windows pipe error - try accessing /docs and using the interactive API",
                "details": str(e)
            }
        raise
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
