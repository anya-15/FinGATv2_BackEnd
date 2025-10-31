"""
Predictor - Top-K Stock Prediction using FinGAT
"""

import torch
from torch_geometric.data import Data
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from pathlib import Path
import pandas as pd
import sys

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.core.model_loader import model_loader
from app.db.models import Stock
from app.config import settings
from data.data_loader import FinancialDataset


class TopKPredictor:
    """
    Predicts top K profitable stocks using FinGAT model
    """
    
    def __init__(self):
        """Initialize predictor - model is loaded lazily on first prediction"""
        self.model = None
        self.metadata = None
        self.data_loader = None
        self.device = torch.device(settings.DEVICE)
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization - only loads model when first needed"""
        if self._initialized:
            return
        
        print("🔧 Initializing predictor...")
        
        # Load model
        try:
            self.model, self.metadata = model_loader.get_model()
            print("✅ Model loaded in predictor")
        except RuntimeError:
            print("⚠️ Model not loaded yet - will load on first prediction")
            # Don't fail - will try again on first prediction
            return
        
        # Initialize data loader
        self.data_loader = FinancialDataset(
            csv_folder_path=settings.DATA_PATH,
            max_stocks=550
        )
        
        self._initialized = True
        print("✅ Predictor initialized with FinancialDataset")
    
    @torch.no_grad()
    def predict_top_k(
        self, 
        db: Session, 
        k: int = 10, 
        sector: Optional[str] = None
    ) -> List[Dict]:
        """
        Get top K profitable stocks with movement predictions
        
        Args:
            db: Database session
            k: Number of top stocks to return
            sector: Optional sector filter (e.g., "Technology")
            
        Returns:
            List of predictions with rank, ticker, company name, movement, etc.
        """
        
        # Ensure initialized
        self._ensure_initialized()
        
        # If still not initialized (no model), try loading again
        if not self._initialized:
            try:
                self.model, self.metadata = model_loader.get_model()
                self.data_loader = FinancialDataset(
                    csv_folder_path=settings.DATA_PATH,
                    max_stocks=550
                )
                self._initialized = True
            except Exception as e:
                print(f"❌ Cannot initialize predictor: {e}")
                return []
        
        # Step 1: Prepare data using your FinancialDataset
        try:
            data, metadata = self.data_loader.prepare_dataset()
        except Exception as e:
            print(f"⚠️ Error preparing dataset: {e}")
            return []
        
        # Step 2: Run inference through FinGAT
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        
        # Your model returns: (returns_pred, movement_pred, ranking_scores, embeddings)
        returns_pred, movement_pred, ranking_scores, embeddings = self.model(x, edge_index)
        
        # Step 3: Get top K based on ranking scores
        ranking_scores = ranking_scores.squeeze()
        
        # Apply sector filter if requested
        if sector and sector.lower() != "all":
            sector_mask = self._get_sector_mask(data, metadata, sector, db)
            ranking_scores = ranking_scores * sector_mask
        
        k_actual = min(k, len(ranking_scores))
        top_k_values, top_k_indices = torch.topk(ranking_scores, k=k_actual)
        
        # Step 4: Get movement predictions
        movement_probs = torch.softmax(movement_pred, dim=1)
        movement_predictions = torch.argmax(movement_pred, dim=1)
        
        # Step 5: Format results
        results = []
        ticker_to_idx = metadata.get('ticker_to_idx', {})
        idx_to_ticker = {v: k for k, v in ticker_to_idx.items()}
        
        for rank, idx in enumerate(top_k_indices, 1):
            idx = idx.item()
            
            # Get ticker from metadata
            ticker = idx_to_ticker.get(idx, f"STOCK_{idx}")
            
            # Find stock info from database
            stock = db.query(Stock).filter(Stock.ticker == ticker).first()
            
            if not stock:
                # If not in DB, try to get name from CSV
                company_name = ticker
            else:
                company_name = stock.company_name
            
            # Get predictions
            predicted_return = returns_pred[idx].item()
            movement_class = movement_predictions[idx].item()
            movement_confidence = movement_probs[idx][movement_class].item()
            ranking_score = top_k_values[rank-1].item()
            
            # Determine movement direction
            predicted_movement = "up" if movement_class == 1 else "down"
            
            # Calculate movement percentage
            movement_percentage = abs(predicted_return * 100)
            
            # Get the date of the price from CSV
            price_date = "2025-10-29"  # Default to yesterday
            latest_price = 0.0
            
            csv_path = Path(settings.DATA_PATH) / f"{ticker}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if 'Date' in df.columns:
                        latest_date_str = str(df['Date'].iloc[-1])
                        # Extract just the date part (YYYY-MM-DD)
                        price_date = latest_date_str.split()[0]
                    
                    if 'Close' in df.columns:
                        latest_price = float(df['Close'].iloc[-1])
                except Exception as e:
                    print(f"⚠️ Error reading {ticker}.csv: {e}")
                    latest_price = float(stock.current_price) if stock and stock.current_price else 0.0
            else:
                latest_price = float(stock.current_price) if stock and stock.current_price else 0.0
            
            results.append({
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
            })
        
        return results
    
    def _get_sector_mask(
        self, 
        data: Data, 
        metadata: Dict, 
        sector: str,
        db: Session
    ) -> torch.Tensor:
        """
        Create mask for filtering stocks by sector
        """
        num_stocks = data.num_nodes
        mask = torch.zeros(num_stocks, device=self.device)
        
        ticker_to_idx = metadata.get('ticker_to_idx', {})
        
        # Get all tickers in requested sector from database
        sector_stocks = db.query(Stock).filter(Stock.sector == sector).all()
        
        for stock in sector_stocks:
            if stock.ticker in ticker_to_idx:
                idx = ticker_to_idx[stock.ticker]
                mask[idx] = 1.0
        
        return mask


# ============================================
# LAZY GLOBAL INSTANCE
# ============================================
# Create instance but don't initialize yet
_predictor_instance = None

def get_predictor() -> TopKPredictor:
    """Get or create predictor instance (lazy loading)"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = TopKPredictor()
    return _predictor_instance
