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


# ====================
# UTILITY ROUTES
# ====================

@router.post("/utils/train")
async def trigger_training(background_tasks: BackgroundTasks):
    """
    Trigger model training with RL optimization.
    This runs the full training pipeline in the background.
    """
    try:
        def run_training():
            import subprocess
            import sys
            result = subprocess.run(
                [sys.executable, "scripts/train_with_hybrid_rl.py"],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            if result.returncode != 0:
                logger.error(f"Training failed: {result.stderr}")
            else:
                logger.info(f"Training completed: {result.stdout}")
        
        background_tasks.add_task(run_training)
        return {
            "status": "started",
            "message": "Training initiated in background",
            "note": "This may take 30-60 minutes. Check logs for progress.",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start training: {str(e)}"
        )


@router.post("/utils/populate-db")
async def populate_database(background_tasks: BackgroundTasks):
    """
    Populate database with stock metadata and predictions.
    Runs the populate_db.py script.
    """
    try:
        def run_populate():
            import subprocess
            import sys
            result = subprocess.run(
                [sys.executable, "scripts/populate_db.py"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            if result.returncode != 0:
                logger.error(f"Populate DB failed: {result.stderr}")
            else:
                logger.info(f"Populate DB completed: {result.stdout}")
        
        background_tasks.add_task(run_populate)
        return {
            "status": "started",
            "message": "Database population initiated",
            "note": "Check logs for progress",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to populate database: {str(e)}"
        )


@router.post("/utils/update-data")
async def update_stock_data(background_tasks: BackgroundTasks):
    """
    Update stock data CSVs with latest market data.
    Runs the update_data.py script.
    """
    try:
        def run_update():
            import subprocess
            import sys
            result = subprocess.run(
                [sys.executable, "scripts/update_data.py"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            if result.returncode != 0:
                logger.error(f"Update data failed: {result.stderr}")
            else:
                logger.info(f"Update data completed: {result.stdout}")
        
        background_tasks.add_task(run_update)
        return {
            "status": "started",
            "message": "Data update initiated",
            "note": "Fetching latest stock data. Check logs for progress.",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update data: {str(e)}"
        )


@router.post("/utils/run-predict-now")
async def run_predict_now_script(background_tasks: BackgroundTasks):
    """
    Run the full predict_now.py script to generate prediction CSVs.
    This creates top5, top10, top20 prediction files.
    """
    try:
        def run_script():
            import subprocess
            import sys
            result = subprocess.run(
                [sys.executable, "predict_now.py"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            if result.returncode != 0:
                logger.error(f"Predict now script failed: {result.stderr}")
            else:
                logger.info(f"Predict now completed: {result.stdout}")
        
        background_tasks.add_task(run_script)
        return {
            "status": "started",
            "message": "Prediction script initiated",
            "note": "Generating prediction CSVs. Check predictions/ folder.",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run predict_now script: {str(e)}"
        )


@router.get("/utils/status")
async def get_utils_status():
    """
    Get status of utility scripts and data files.
    """
    import os
    from pathlib import Path
    
    # Check if scripts exist
    scripts_status = {
        "train_script": os.path.exists("scripts/train_with_hybrid_rl.py"),
        "populate_script": os.path.exists("scripts/populate_db.py"),
        "update_script": os.path.exists("scripts/update_data.py"),
        "predict_script": os.path.exists("predict_now.py")
    }
    
    # Check data folders
    csv_count = len(list(Path("indian_data").glob("*.csv"))) if Path("indian_data").exists() else 0
    prediction_count = len(list(Path("predictions").glob("*.csv"))) if Path("predictions").exists() else 0
    checkpoint_count = len(list(Path("checkpoints").glob("*.ckpt"))) if Path("checkpoints").exists() else 0
    
    return {
        "scripts": scripts_status,
        "data": {
            "csv_files": csv_count,
            "prediction_files": prediction_count,
            "checkpoints": checkpoint_count
        },
        "timestamp": datetime.now().isoformat()
    }


@router.get("/predict/single/{ticker}")
async def predict_single_stock(ticker: str):
    """
    Get prediction for a single stock from pre-generated CSV files.
    
    Args:
        ticker: Stock ticker symbol (e.g., INFY, TCS, RELIANCE)
    
    Returns:
        Detailed prediction with confidence, direction, rank, and trading suggestions
    """
    try:
        # Import the predictor module
        from scripts.predict_single_stock import SingleStockPredictor
        
        # Initialize predictor
        try:
            predictor = SingleStockPredictor('predictions')
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="No prediction files found. Run POST /api/v1/utils/run-predict-now first to generate predictions."
            )
        
        # Get prediction
        result = predictor.predict_stock(ticker)
        
        if result is None:
            # Find similar tickers
            all_tickers = predictor.predictions_df['Ticker'].tolist()
            similar = [t for t in all_tickers if ticker.upper() in t][:5]
            
            raise HTTPException(
                status_code=404,
                detail=f"Ticker '{ticker.upper()}' not found. Similar tickers: {', '.join(similar) if similar else 'None'}"
            )
        
        # Determine confidence strength
        confidence = result['confidence']
        if confidence > 50:
            confidence_level = "VERY HIGH"
        elif confidence > 35:
            confidence_level = "HIGH"
        elif confidence > 20:
            confidence_level = "MODERATE"
        else:
            confidence_level = "LOW"
        
        # Determine quality based on rank
        rank = result['rank']
        
        if rank <= 5:
            quality = "EXCELLENT (Top 5!)"
        elif rank <= 10:
            quality = "EXCELLENT (Top 10!)"
        elif rank <= 20:
            quality = "VERY GOOD (Top 20)"
        elif rank <= 50:
            quality = "GOOD (Top 50)"
        else:
            quality = "AVERAGE"
        
        # Trading suggestion
        direction = result['direction']
        if direction == "UP" and confidence > 35 and rank <= 20:
            suggestion = "STRONG BUY"
            allocation = "10-15%"
            risk_level = "LOW"
        elif direction == "UP" and confidence > 20:
            suggestion = "BUY"
            allocation = "5-10%"
            risk_level = "MODERATE"
        elif direction == "UP":
            suggestion = "WEAK BUY"
            allocation = "2-5%"
            risk_level = "HIGH"
        elif direction == "DOWN" and confidence > 35:
            suggestion = "STRONG SELL/AVOID"
            allocation = "0%"
            risk_level = "HIGH"
        elif direction == "DOWN" and confidence > 20:
            suggestion = "SELL/AVOID"
            allocation = "0%"
            risk_level = "MODERATE"
        else:
            suggestion = "NEUTRAL"
            allocation = "0-2%"
            risk_level = "HIGH"
        
        # Calculate expected profits for different amounts
        expected_return = result['expected_return']
        investment_scenarios = []
        if direction == "UP" and rank <= 50:
            for amount in [10000, 50000, 100000, 200000]:
                expected_profit = amount * (expected_return / 100)
                investment_scenarios.append({
                    "investment": amount,
                    "expected_profit": round(expected_profit, 2),
                    "expected_value": round(amount + expected_profit, 2)
                })
        
        return {
            "ticker": result['ticker'],
            "sector": result['sector'],
            "prediction_date": predictor.prediction_date,
            "prediction": {
                "direction": direction,
                "confidence_percentage": round(confidence, 2),
                "confidence_level": confidence_level,
                "up_probability": round(result['up_probability'], 2),
                "expected_return_percentage": round(expected_return, 2),
                "ranking_score": round(result['ranking_score'], 4),
                "rank": rank,
                "total_stocks": result['total_stocks'],
                "quality": quality
            },
            "trading_suggestion": {
                "action": suggestion,
                "suggested_allocation": allocation,
                "risk_level": risk_level
            },
            "investment_scenarios": investment_scenarios,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting prediction: {str(e)}"
        )
