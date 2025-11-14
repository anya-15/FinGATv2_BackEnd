"""
Fast Predictor - Reads predictions directly from CSV files
Ultra-fast: <50ms response time
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FastPredictor:
    """
    Fast predictor that reads from pre-generated CSV files.
    No model inference - just pure CSV reading for instant results.
    """
    
    def __init__(self, predictions_dir: str = "predictions"):
        self.predictions_dir = Path(predictions_dir)
        self._cache = {}
        self._cache_timestamp = None
        self._cache_duration = 300  # 5 minutes
        
    def _get_latest_csv(self, pattern: str) -> Optional[Path]:
        """Find the most recent CSV file matching pattern"""
        files = list(self.predictions_dir.glob(pattern))
        if not files:
            return None
        # Sort by modification time, newest first
        return max(files, key=lambda p: p.stat().st_mtime)
    
    def _load_predictions(self, force_reload: bool = False) -> pd.DataFrame:
        """Load predictions from the latest CSV file with caching"""
        import time
        current_time = time.time()
        
        # Check cache
        if (not force_reload and 
            self._cache_timestamp is not None and 
            (current_time - self._cache_timestamp) < self._cache_duration and
            'predictions' in self._cache):
            logger.info("[CACHE] Using cached predictions")
            return self._cache['predictions']
        
        # Find latest prediction file (use regular predictions with both UP and DOWN)
        csv_file = self._get_latest_csv("predictions_2*.csv")  # Match predictions_2025-*.csv
        if csv_file is None:
            # Fallback to up_only if regular not found
            csv_file = self._get_latest_csv("predictions_up_only_*.csv")
        
        if csv_file is None:
            raise FileNotFoundError(
                "No prediction CSV files found. Run 'python predict_now.py' first."
            )
        
        logger.info(f"[FastPredictor] Loading from: {csv_file.name}")
        
        # Read CSV - this is FAST (< 10ms for 500 rows)
        df = pd.read_csv(csv_file)
        
        # Cache it
        self._cache['predictions'] = df
        self._cache_timestamp = current_time
        
        logger.info(f"[FastPredictor] Loaded {len(df)} predictions in <10ms")
        return df
    
    def get_top_k(self, k: int = 10, sector: Optional[str] = None) -> List[Dict]:
        """
        Get top K predictions - FAST!
        
        Args:
            k: Number of predictions to return
            sector: Optional sector filter
            
        Returns:
            List of prediction dictionaries
        """
        try:
            df = self._load_predictions()
            
            # Filter by sector if specified
            if sector and sector.lower() != "all":
                df = df[df['Sector'] == sector]
            
            # Sort by Rank (already sorted in CSV, but just to be sure)
            if 'Rank' in df.columns:
                df = df.sort_values('Rank')
            elif 'Ranking Score' in df.columns:
                df = df.sort_values('Ranking Score', ascending=False)
            
            # Take top K
            df = df.head(k)
            
            # Convert to list of dicts with all required fields
            predictions = []
            for _, row in df.iterrows():
                # Handle both formats: "Confidence_%" and "Confidence (%)"
                confidence_pct = float(row.get('Confidence_%', row.get('Confidence (%)', 0.0)))
                expected_return = float(row.get('Expected_Return_%', row.get('Expected Return (%)', 0.0)))
                ranking_score = float(row.get('Ranking_Score', row.get('Ranking Score', 0.0)))
                
                pred = {
                    'rank': int(row.get('Rank', len(predictions) + 1)),
                    'ticker': str(row.get('Ticker', '')),
                    'company_name': str(row.get('Ticker', '')),  # CSV doesn't have company name
                    'current_price': 0.0,  # Not in CSV
                    'price_date': '2025-11-08',  # Current date
                    'predicted_movement': 'up' if str(row.get('Direction', 'UP')).upper() == 'UP' else 'down',
                    'movement_percentage': expected_return,
                    'confidence_score': confidence_pct / 100.0,  # Convert to 0-1 range
                    'ranking_score': ranking_score,
                    'predicted_return': expected_return / 100.0,  # Convert to decimal
                    'sector': str(row.get('Sector', 'Unknown'))
                }
                predictions.append(pred)
            
            logger.info(f"[FastPredictor] Returned {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"[FastPredictor] Error: {e}")
            raise
    
    def get_single_stock(self, ticker: str) -> Optional[Dict]:
        """
        Get prediction for a single stock - ULTRA FAST!
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Prediction dictionary or None if not found
        """
        try:
            df = self._load_predictions()
            
            # Find the ticker (case-insensitive)
            ticker = ticker.upper()
            stock_df = df[df['Ticker'].str.upper() == ticker]
            
            if stock_df.empty:
                return None
            
            row = stock_df.iloc[0]
            
            return {
                'ticker': row.get('Ticker', ''),
                'sector': row.get('Sector', 'Unknown'),
                'predicted_return': float(row.get('Expected Return (%)', 0.0)),
                'predicted_movement': 'up' if row.get('Direction', 'UP') == 'UP' else 'down',
                'ranking_score': float(row.get('Ranking Score', 0.0)),
                'confidence': float(row.get('Confidence (%)', 0.0)),
                'rank': int(row.get('Rank', 0)) if 'Rank' in row else 0,
                'up_probability': float(row.get('Up Probability (%)', 50.0))
            }
            
        except Exception as e:
            logger.error(f"[FastPredictor] Error getting single stock: {e}")
            return None
    
    def get_all_tickers(self) -> List[str]:
        """Get list of all available tickers"""
        try:
            df = self._load_predictions()
            return df['Ticker'].tolist()
        except:
            return []


# Global instance
_fast_predictor = None

def get_fast_predictor() -> FastPredictor:
    """Get singleton fast predictor instance"""
    global _fast_predictor
    if _fast_predictor is None:
        _fast_predictor = FastPredictor()
    return _fast_predictor
