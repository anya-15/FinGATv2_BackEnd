"""
API Response Schemas - Pydantic Models
Defines the structure of JSON responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class StockPrediction(BaseModel):
    """Single stock prediction"""
    rank: int = Field(..., description="Rank in top-K list (1 = best)")
    ticker: str = Field(..., description="Stock ticker symbol (e.g., RELIANCE)")
    company_name: str = Field(..., description="Full company name")
    current_price: float = Field(..., description="Current stock price")
    price_date: str = Field(..., description="Date of price (YYYY-MM-DD)")
    predicted_movement: str = Field(..., description="Predicted direction: 'up' or 'down'")
    movement_percentage: float = Field(..., description="Expected movement percentage")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence (0-1)")
    ranking_score: float = Field(..., description="Profitability ranking score")
    predicted_return: float = Field(..., description="Expected return value")
    sector: str = Field(..., description="Stock sector (e.g., Technology, Finance)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "rank": 1,
                "ticker": "RELIANCE",
                "company_name": "Reliance Industries Ltd",
                "current_price": 2450.75,
                "price_date": "2025-10-29",
                "predicted_movement": "up",
                "movement_percentage": 3.45,
                "confidence_score": 0.87,
                "ranking_score": 0.92,
                "predicted_return": 0.0345,
                "sector": "Energy"
            }
        }


class TopKResponse(BaseModel):
    """Response for top-K predictions endpoint"""
    timestamp: datetime = Field(..., description="Prediction timestamp")
    k: int = Field(..., description="Number of stocks requested")
    sector: Optional[str] = Field(None, description="Sector filter applied")
    total_stocks_analyzed: int = Field(..., description="Total stocks in dataset")
    predictions: List[StockPrediction] = Field(..., description="List of top K stock predictions")
    model_version: str = Field(..., description="Model version used")
    model_type: str = Field(..., description="Model architecture type")
    prediction_horizon_days: int = Field(..., description="Days ahead for prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-10-30T10:35:00",
                "k": 10,
                "sector": "Technology",
                "total_stocks_analyzed": 148,
                "predictions": [
                    {
                        "rank": 1,
                        "ticker": "TCS",
                        "company_name": "Tata Consultancy Services",
                        "current_price": 3450.50,
                        "price_date": "2025-10-29",
                        "predicted_movement": "up",
                        "movement_percentage": 4.2,
                        "confidence_score": 0.89,
                        "ranking_score": 0.94,
                        "predicted_return": 0.042,
                        "sector": "Technology"
                    }
                ],
                "model_version": "FinGATv2",
                "model_type": "Graph Attention Network",
                "prediction_horizon_days": 7
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: datetime = Field(..., description="Current timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "timestamp": "2025-10-30T10:35:00"
            }
        }


class ModelStatusResponse(BaseModel):
    """Model training status response"""
    model_exists: bool = Field(..., description="Whether trained model exists")
    last_trained: Optional[datetime] = Field(None, description="Last training timestamp")
    checkpoint_path: str = Field(..., description="Path to model checkpoint")
    model_loaded: bool = Field(..., description="Whether model is loaded in memory")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_exists": True,
                "last_trained": "2025-10-27T18:30:00",
                "checkpoint_path": "checkpoints/production_model.ckpt",
                "model_loaded": True
            }
        }
