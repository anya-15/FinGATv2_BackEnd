"""
FinGAT Backend API - Main Application
FastAPI server with daily automated training
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import settings
from app.db.database import init_db
from app.core.model_loader import model_loader
from app.api.routes import router
from app.scheduler.tasks import setup_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events
    Runs when API starts and stops
    """
    print("=" * 60)
    print("🚀 Starting FinGAT API - Indian Stock Predictions")
    print("=" * 60)
    
    # Initialize database
    print("📊 Initializing database...")
    init_db()
    
    # Load model (or train if doesn't exist)
    print("🧠 Loading FinGAT model...")
    try:
        model_loader.load_model()
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print("⚠️ No trained model found")
        print("💡 Run: python scripts/train_model.py")
        print("   Or trigger training via: POST /api/v1/retrain")
    except Exception as e:
        print(f"⚠️ Model loading error: {e}")
    
    # Start daily training scheduler
    print("⏰ Starting daily training scheduler...")
    scheduler = setup_scheduler()
    scheduler.start()
    print(f"   Scheduled for: {settings.TRAINING_HOUR}:{settings.TRAINING_MINUTE:02d} {settings.TRAINING_TIMEZONE}")
    
    print("=" * 60)
    print("✅ FinGAT API is ready!")
    print(f"📖 API Docs: http://localhost:{settings.API_PORT}/docs")
    print(f"🔍 Health: http://localhost:{settings.API_PORT}/api/v1/health")
    print("=" * 60)
    
    yield
    
    # Shutdown
    print("\n🛑 Shutting down FinGAT API...")
    scheduler.shutdown()
    print("✅ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="2.0.0",
    description="""
    **FinGAT - Graph Attention Network for Indian Stock Market Prediction**
    
    Features:
    - 📈 Top-K profitable stock predictions (NSE/BSE)
    - 🤖 Daily automated model retraining
    - 📊 Movement direction prediction (Up/Down)
    - 🎯 Ranking scores with confidence levels
    - 🏦 Sector-based filtering
    
    Powered by PyTorch Geometric + Lightning ⚡
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - allows frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: ["https://your-frontend.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root endpoint
@app.get("/", tags=["Root"])
def root():
    """API root endpoint"""
    return {
        "message": "FinGAT Stock Prediction API",
        "version": "2.0.0",
        "market": "Indian Stocks (NSE/BSE)",
        "model": "Graph Attention Network v2",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/api/v1/health",
            "predictions": "/api/v1/predict/top-k",
            "sectors": "/api/v1/sectors",
            "retrain": "/api/v1/retrain"
        }
    }


# Include API routes
app.include_router(
    router, 
    prefix=settings.API_V1_PREFIX, 
    tags=["predictions"]
)


# Run with: uvicorn app.main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=True
    )
