"""
API Routes - All endpoints for FinGAT API
"""

from fastapi import APIRouter, Depends, Query, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime
import os

from app.db.database import get_db
from app.db.models import Stock
from app.core.predictor import get_predictor, run_predict_now  # <-- Make sure run_predict_now() exists!
from app.core.model_loader import model_loader
from app.schemas.responses import (
    TopKResponse,
    StockPrediction,
    HealthResponse,
    ModelStatusResponse
)

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns API status and model loading status
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.is_loaded(),
        timestamp=datetime.now()
    )

@router.get("/predict/top-k", response_model=TopKResponse)
async def get_top_k_stocks(
    k: int = Query(10, ge=1, le=50, description="Number of top stocks to return"),
    sector: Optional[str] = Query(None, description="Filter by sector (e.g., Technology, Finance)"),
    db: Session = Depends(get_db)
):
    """
    Get top K profitable stocks with movement predictions
    """
    try:
        predictor = get_predictor()
        predictions = predictor.predict_top_k(db, k, sector)
        if len(predictions) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No stocks found" + (f" in sector '{sector}'" if sector else "")
            )
        return TopKResponse(
            timestamp=datetime.now(),
            k=k,
            sector=sector,
            total_stocks_analyzed=len(predictions),
            predictions=predictions,
            model_version="FinGATv2",
            model_type="Graph Attention Network",
            prediction_horizon_days=7
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@router.get("/sectors")
async def get_sectors(db: Session = Depends(get_db)):
    """
    Get list of available stock sectors
    """
    try:
        sectors = db.query(Stock.sector).distinct().all()
        sector_list = ["All"] + sorted([s[0] for s in sectors if s[0]])
        return {
            "sectors": sector_list,
            "total": len(sector_list) - 1  # Exclude "All"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch sectors: {str(e)}"
        )

@router.get("/stocks")
async def list_stocks(
    sector: Optional[str] = Query(None, description="Filter by sector"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum stocks to return"),
    db: Session = Depends(get_db)
):
    """
    Get list of available stocks
    """
    try:
        query = db.query(Stock)
        if sector and sector.lower() != "all":
            query = query.filter(Stock.sector == sector)
        stocks = query.limit(limit).all()
        return {
            "total": len(stocks),
            "stocks": [
                {
                    "ticker": s.ticker,
                    "company_name": s.company_name,
                    "sector": s.sector,
                    "current_price": float(s.current_price) if s.current_price else 0.0
                }
                for s in stocks
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch stocks: {str(e)}"
        )

@router.get("/model/info")
async def get_model_info():
    """
    Get model information and configuration
    """
    from app.config import model_config
    try:
        model, metadata = model_loader.get_model()
        return {
            "model_name": model_config.get('model', {}).get('name', 'FinGAT'),
            "version": "v2.0",
            "architecture": "Graph Attention Network (GATv2)",
            "framework": "PyTorch Lightning + torch-geometric",
            "market": "Indian Stocks (NSE/BSE)",
            "features": [
                "stock ranking",
                "movement prediction",
                "return forecasting"
            ],
            "prediction_horizon": "7 days",
            "model_config": {
                "hidden_dim": model_config.get('model', {}).get('hidden_dim'),
                "num_heads": model_config.get('model', {}).get('num_heads'),
                "num_layers": model_config.get('model', {}).get('num_layers'),
                "dropout": model_config.get('model', {}).get('dropout')
            },
            "training_stocks": metadata.get("num_stocks", "Unknown") if metadata else "Unknown",
            "num_features": metadata.get("num_features", "Unknown") if metadata else "Unknown"
        }
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first."
        )

@router.get("/model/status", response_model=ModelStatusResponse)
async def get_model_status():
    """
    Get current model training status and last training time
    """
    checkpoint_path = "checkpoints/production_model.ckpt"
    if os.path.exists(checkpoint_path):
        last_modified = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
        return ModelStatusResponse(
            model_exists=True,
            last_trained=last_modified,
            checkpoint_path=checkpoint_path,
            model_loaded=model_loader.is_loaded()
        )
    else:
        return ModelStatusResponse(
            model_exists=False,
            last_trained=None,
            checkpoint_path=checkpoint_path,
            model_loaded=False
        )

@router.post("/retrain")
async def trigger_manual_retrain(background_tasks: BackgroundTasks):
    """
    Manually trigger model retraining
    """
    from app.scheduler.tasks import daily_pipeline
    background_tasks.add_task(daily_pipeline)
    return {
        "status": "Training started",
        "message": "Model retraining initiated in background",
        "note": "Check server logs for progress",
        "timestamp": datetime.now().isoformat()
    }


# ====================
# NEW: Full Predict Now Endpoint!
# ====================

@router.get("/predict/now")
async def predict_now_full():
    """
    Returns full FinGAT RL "predict now" batch results (top-K, confidence, sector, etc.)
    """
    try:
        result = run_predict_now()  # This should run your full RL pipeline and return JSON serializable output
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Predict-now batch failed: {str(e)}"
        )

@router.post("/reload-features")
async def reload_feature_mask():
    """
    Reload feature mask from latest training run.
    Call this after training a new model to immediately use the new features without restarting the server.
    """
    try:
        predictor = get_predictor()
        predictor.reload_feature_mask()
        
        # Get info about loaded mask
        if predictor.feature_mask is not None:
            num_features = int(predictor.feature_mask.sum())
            total_features = len(predictor.feature_mask)
            mask_list = predictor.feature_mask.cpu().numpy().tolist()
        else:
            num_features = "all"
            total_features = "unknown"
            mask_list = None
        
        return {
            "status": "success",
            "message": "Feature mask reloaded from latest training run",
            "features_selected": num_features,
            "total_features": total_features,
            "feature_mask": mask_list,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload feature mask: {str(e)}"
        )
