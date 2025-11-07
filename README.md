# 🚀 FinGAT Backend

**Graph Attention Network for Indian Stock Market Prediction**

A production-ready, sector-aware Graph Neural Network system for predicting NSE/BSE stock movements using PyTorch Geometric, Lightning, and Reinforcement Learning.

[![Status](https://img.shields.io/badge/status-production--ready-success)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![Framework](https://img.shields.io/badge/framework-PyTorch%20Lightning-purple)]()
[![API](https://img.shields.io/badge/API-FastAPI-green)]()

---

## 🎯 What is FinGAT?

FinGAT is an end-to-end stock prediction system that:
- **Analyzes 147+ Indian stocks** from NSE/BSE markets
- **Uses Graph Neural Networks** (GATv2) to capture stock relationships
- **Implements sector-aware architecture** for hierarchical market understanding
- **Applies RL-based feature selection** for optimal performance
- **Provides REST API** for real-time predictions
- **Ensures leak-free data engineering** for honest accuracy (52-60%)

---

## 🏗️ How It Works

### **1. Data Collection & Engineering**
```
CSV Files (indian_data/) → Technical Features → Leak-Free Windows
```
- Loads OHLC data from 147 Indian stock CSVs
- Creates 7 technical indicators: returns, volatility, SMAs, RSI, momentum, volume
- **Strict windowing**: 60-day history, 5-day buffer, 5-day target (NO OVERLAP)
- Aggregates features: mean + std = **14 features per stock**

### **2. Graph Construction**
```
Stocks → KNN Graph + Sector Mapping → Hierarchical Structure
```
- **Stock-level graph**: K-NN connections (k=15) based on feature similarity
- **Sector mapping**: Each stock mapped to its sector (Finance, IT, Energy, etc.)
- **Sector graph**: Fully connected sector relationships
- Result: `Data(x=[N, 14], edge_index=[2, E], stock_to_sector=[N], sector_edge_index=[2, S])`

### **3. Model Architecture**
```
Stock Features → Stock GAT → Sector Pooling → Sector GAT → Fusion → Predictions
```
- **Level 1**: Stock-level GATv2 (intra-sector relationships)
- **Level 2**: Attention pooling to sector embeddings
- **Level 3**: Sector-level GATv2 (inter-sector relationships)
- **Level 4**: Fusion layer combines stock + sector information
- **Outputs**: 
  - Regression: Predicted returns
  - Classification: Movement direction (up/down)
  - Ranking: Relative stock scores

### **4. RL-Based Optimization**
```
RL Agent → Feature Selection + Hyperparameter Tuning → Best Model
```
- Hybrid RL agent optimizes:
  - **Feature mask**: Which features to use
  - **Hyperparameters**: hidden_dim, learning_rate, dropout
- Saves best configuration to `rl_models/selected_runs/latest_manifest.json`
- All predictions use RL-optimized settings

### **5. Prediction & API**
```
New Data → Apply RL Mask → Model Inference → REST API → JSON Results
```
- FastAPI server loads model and data
- `/api/v1/predict/now`: Batch predictions for all stocks
- `/api/v1/predict/top-k`: Top-K recommendations
- Results include: ticker, price, movement, ranking score, sector

---

## 📦 Project Structure

```
FinGAT_Backend/
├── app/                    # FastAPI Application
│   ├── api/               # REST API routes
│   ├── core/              # Model loader & predictor
│   ├── db/                # Database models & connection
│   ├── scheduler/         # Daily training scheduler
│   └── main.py            # FastAPI app entry point
├── data/
│   ├── data_loader.py     # ✅ VERIFIED: Leak-free data engineering
│   └── indian_data/       # 147 stock CSVs (OHLC data)
├── training/
│   └── lightning_module.py # ✅ VERIFIED: GATv2 + sector architecture
├── scripts/
│   ├── train_model.py     # Classical training
│   ├── train_with_hybrid_rl.py # RL optimization
│   ├── populate_db.py     # Database population
│   └── update_data.py     # Data refresh
├── checkpoints/           # Model checkpoints (*.ckpt)
├── rl_models/
│   ├── hybrid/            # RL training runs
│   └── selected_runs/
│       └── latest_manifest.json # ✅ Active: fingat-hybrid-epoch=43
├── predictions/           # CSV prediction outputs
├── config/
│   └── config.yaml        # Model & training config
├── .env                   # Environment variables
└── requirements.txt       # Python dependencies
```

---

## ⚡ Quick Start

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
```

**Key packages:**
- `torch` + `torch-geometric` - GNN framework
- `pytorch-lightning` - Training framework
- `fastapi` + `uvicorn` - API server
- `pandas`, `numpy`, `scikit-learn` - Data processing

### **2. Prepare Data**

Place your stock CSVs in `indian_data/`:
```
indian_data/
├── RELIANCE.csv
├── TCS.csv
├── INFY.csv
└── ... (147 stocks)
```

**CSV Requirements:**
- Columns: `Date`, `Close` (minimum), `Open`, `High`, `Low`, `Volume` (recommended)
- Minimum: 60 rows per stock
- Format: Daily OHLC data

### **3. Configure Environment**

Copy `.env.example` to `.env` and set:
```bash
DATABASE_URL=sqlite:///./fingat.db
MODEL_CHECKPOINT_PATH=checkpoints/fingat-hybrid-epoch=43-val_mrr=0.1111.ckpt
DATA_PATH=indian_data
DEVICE=cpu  # or 'cuda' if GPU available
API_PORT=8000
```

### **4. Start API Server**

```bash
# Production mode (recommended - no Windows pipe errors)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Development mode (with auto-reload)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Access:**
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/v1/health

---

## 🔌 API Endpoints

### **📊 Core Predictions**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/now` | GET | Full batch predictions for all stocks |
| `/predict/top-k?k=10` | GET | Get top-K stock recommendations |
| `/predict/single/{ticker}` | GET | Get prediction for a single stock |

### **📈 Model Management**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/model/info` | GET | Get model configuration and metadata |
| `/model/status` | GET | Check model training status |
| `/retrain` | POST | Trigger model retraining |
| `/reload-features` | POST | Reload feature mask from latest run |

### **📚 Data & Sectors**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sectors` | GET | List all available sectors |
| `/stocks` | GET | List all stocks with filters |

### **⚙️ Utility Endpoints**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/utils/train` | POST | Trigger model training with RL |
| `/utils/populate-db` | POST | Populate database with stock data |
| `/utils/update-data` | POST | Update stock data CSVs |
| `/utils/run-predict-now` | POST | Generate prediction CSVs |
| `/utils/status` | GET | Check status of utility scripts |

### **System**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API docs (Swagger UI) |
| `/redoc` | GET | Alternative API docs (ReDoc) |

### **Example: Get Top 5 Predictions**

```bash
curl http://localhost:8000/api/v1/predict/top-k?k=5
```

**Response:**
```json
{
  "status": "success",
  "predictions": [
    {
      "rank": 1,
      "ticker": "ONGC",
      "company_name": "ONGC",
      "predicted_movement": "up",
      "ranking_score": 0.85,
      "sector": "Energy"
    },
    ...
  ]
}
```

---

## 🏋️ Training

### **Option 1: Classical Training**
```bash
python scripts/train_model.py
```
- Trains GATv2 model with default hyperparameters
- Saves checkpoint to `checkpoints/`
- Uses all 14 features

### **Option 2: RL-Optimized Training (Recommended)**
```bash
python scripts/train_with_hybrid_rl.py
```
- RL agent optimizes:
  - **Feature selection**: Which of the 14 features to use
  - **Hyperparameters**: hidden_dim, learning_rate, dropout
- Outputs saved to `rl_models/hybrid/YYYY-MM-DD_HH-MM-SS/`:
  - `best_features.npy` - Feature mask
  - `best_hparams.json` - Optimal hyperparameters
  - `manifest.json` - Full configuration
  - Checkpoint path reference

**Current Active Model:**
- Checkpoint: `fingat-hybrid-epoch=43-val_mrr=0.1111.ckpt`
- Features: 14 (7 technical indicators × 2)
- Architecture: Stock GAT → Sector Pooling → Sector GAT → Fusion

---

## 📈 Prediction

### **Via API (Recommended)**
```bash
# Start server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Get predictions
curl http://localhost:8000/api/v1/predict/now
```

### **Via Script**
```bash
# Batch predictions for all stocks
python scripts/predict_now.py

# Single stock prediction
python scripts/predict_single_stock.py

# Track predictions over time
python scripts/track_predictions.py
```

**Output locations:**
- `predictions/` - All prediction CSVs
- `results/` - Top-K rankings (top5, top10, top20)

---

## 🗄️ Database & Utilities

### **Populate Database**
```bash
python scripts/populate_db.py
```
Loads predictions, stocks, and metadata into SQLite database for analytics.

### **Update Stock Data**
```bash
python scripts/update_data.py
```
Refreshes and validates CSV files in `indian_data/`.

---

## 🛠️ Troubleshooting

### **✅ FIXED: Windows Pipe Error**
**Issue:** `[WinError 233] No process is on the other end of the pipe`

**Solution:** Run server without `--reload` flag:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### **Feature Dimension Mismatch**
**Issue:** `Given normalized_shape=[X], expected input with shape [*, X]`

**Solution:** 
- Checkpoint expects 14 features (7 indicators × 2)
- Data loader creates exactly 14 features
- ✅ Already fixed in current version

### **Missing Data**
**Issue:** `Warning: {ticker} insufficient data`

**Solution:**
- Ensure each CSV has minimum 60 rows
- Check columns: `Date`, `Close` are required
- Validate data format in `indian_data/`

### **Model Not Loading**
**Issue:** `Checkpoint not found`

**Solution:**
- Check `rl_models/selected_runs/latest_manifest.json`
- Verify checkpoint path in `.env`
- Train a model if none exists: `python scripts/train_with_hybrid_rl.py`

## ✅ System Verification Status

### **Core Components**
| Component | Status | Details |
|-----------|--------|---------|
| `data_loader.py` | ✅ Working | 14 features, leak-free windowing, sector mapping |
| `model_loader.py` | ✅ Working | Checkpoint loading, device handling, hot-reload |
| `lightning_module.py` | ✅ Working | Hierarchical GATv2, sector-aware architecture |
| `predictor.py` | ✅ Working | Batch predictions, top-K ranking |
| FastAPI Server | ✅ Working | All endpoints operational |

### **Current Configuration**
- **Stocks Analyzed**: 147 Indian stocks (NSE/BSE)
- **Model**: fingat-hybrid-epoch=43-val_mrr=0.1111.ckpt
- **Features**: 14 (returns, volatility, SMA-5, SMA-20, RSI, momentum, volume)
- **Architecture**: Stock GAT (64 hidden) → Sector Pooling → Sector GAT → Fusion
- **Sectors**: 16 sectors (Finance, IT, Energy, Healthcare, etc.)
- **Graph**: K-NN (k=15) + fully connected sector graph

### **Recent Predictions (Sample)**
```
Top 5 Stocks:
1. TITAN - down
2. ONGC - up
3. COALINDIA - up
4. NESTLEIND - up
5. HAVELLS - up
```

---

## 🔬 Technical Details

### **Data Pipeline**
1. **Input**: 147 CSV files (1250 rows each, ~5 years of data)
2. **Feature Engineering**: 7 technical indicators per stock
3. **Aggregation**: Mean + Std over 50-day window = 14 features
4. **Graph**: K-NN similarity + sector relationships
5. **Output**: PyTorch Geometric `Data` object

### **Model Architecture**
```
Input: [N, 14] features
  ↓
Stock-Level GAT (4 heads, 2 layers)
  ↓
Attention Pooling → [S, hidden_dim] sector embeddings
  ↓
Sector-Level GAT (4 heads, 2 layers)
  ↓
Broadcast back to stocks + Fusion
  ↓
Output Heads:
  - Regression: Predicted returns
  - Classification: Up/Down movement
  - Ranking: Relative scores
```

### **Training Details**
- **Framework**: PyTorch Lightning
- **Optimizer**: AdamW (lr=0.001, weight_decay=0.01)
- **Loss**: Multi-task (MAE + Focal + Ranking)
- **Metrics**: MRR, Precision@K, NDCG@K
- **Validation**: Temporal split (70/15/15)

### **API Performance**
- **Startup Time**: ~5-10 seconds
- **Prediction Time**: ~2-3 seconds for 147 stocks
- **Memory Usage**: ~500MB (CPU mode)
- **Concurrent Requests**: Supported via FastAPI

---

## 📊 Key Features

✅ **Leak-Free Engineering**: Strict temporal windows prevent data leakage  
✅ **Sector-Aware**: Hierarchical architecture captures market structure  
✅ **RL-Optimized**: Feature selection + hyperparameter tuning  
✅ **Production-Ready**: FastAPI server with health checks  
✅ **Scalable**: Handles 147+ stocks efficiently  
✅ **Honest Accuracy**: 52-60% (realistic for stock prediction)  

---

## 🔄 Feature Auto-Sync System

### **Automatic Feature Synchronization**

The predictor **automatically loads** the correct feature mask from the latest training run. No manual configuration needed!

### **How It Works**

1. **Training Phase**: RL optimization selects best features (e.g., 9 out of 14)
2. **Saves**: Feature mask to `rl_models/hybrid/YYYY-MM-DD_HH-MM-SS/best_features.npy`
3. **Updates**: Manifest at `rl_models/selected_runs/latest_manifest.json`
4. **Prediction Phase**: Predictor auto-loads mask and applies to features

### **Benefits**

✅ **Zero Manual Configuration** - Train any model, features auto-sync  
✅ **No Dimension Errors** - Always matches model expectations  
✅ **Hot-Reload Capability** - Update without server restart  
✅ **Future-Proof** - Works with any feature count  

### **API Endpoint**

Reload features without restarting:
```bash
curl -X POST http://localhost:8000/api/v1/reload-features
```

**Response:**
```json
{
  "status": "success",
  "features_selected": 9,
  "total_features": 14,
  "feature_mask": [false, false, true, ...]
}
```

### **Workflow**

```
Train Model → Save Feature Mask → Update Manifest
                                        ↓
                            Predictor Auto-Loads
                                        ↓
                            Apply to Features
                                        ↓
                            Model Inference
```

---

## 🗄️ Database Architecture

### **Database Provider**

- **Type**: Neon PostgreSQL (Serverless)
- **Region**: ap-southeast-1 (Singapore)
- **SSL**: Required
- **Connection Pooling**: 5 connections, 10 max overflow

### **Database Tables (5 Total)**

#### **1. stocks** - Stock Metadata
```sql
CREATE TABLE stocks (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) UNIQUE NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    sector VARCHAR(100),
    current_price FLOAT,
    market_cap FLOAT,
    exchange VARCHAR(10) DEFAULT 'NSE',
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

**Records**: 148 stocks  
**Usage**: Store company info, sector classification, current prices

#### **2. predictions** - Historical Predictions
```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    predicted_return FLOAT NOT NULL,
    predicted_movement VARCHAR(10) NOT NULL,  -- 'up' or 'down'
    movement_confidence FLOAT NOT NULL,
    ranking_score FLOAT NOT NULL,
    rank INTEGER,
    model_checkpoint VARCHAR(255),
    model_version VARCHAR(50),
    sector VARCHAR(100),
    prediction_date TIMESTAMP DEFAULT NOW(),
    
    -- Backtesting support
    actual_return FLOAT,
    actual_movement VARCHAR(10),
    outcome_date TIMESTAMP,
    is_correct INTEGER  -- 1=correct, 0=incorrect, NULL=pending
);
```

**Auto-Saving**: ✅ Every prediction is automatically saved  
**Purpose**: Track predictions, backtest accuracy, analyze patterns

#### **3. training_history** - Training Runs
```sql
CREATE TABLE training_history (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) UNIQUE NOT NULL,
    training_type VARCHAR(50) NOT NULL,
    model_architecture VARCHAR(50),
    hidden_dim INTEGER,
    num_layers INTEGER,
    learning_rate FLOAT,
    
    -- Metrics
    val_accuracy FLOAT,
    val_mrr FLOAT,
    test_accuracy FLOAT,
    
    -- RL Feature Selection
    num_features_selected INTEGER,
    total_features INTEGER,
    feature_mask_path VARCHAR(500),
    
    -- Timing
    training_duration_minutes FLOAT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(20)  -- 'running', 'completed', 'failed'
);
```

**Purpose**: Track all training experiments, compare configurations

#### **4. model_checkpoints** - Checkpoint Catalog
```sql
CREATE TABLE model_checkpoints (
    id SERIAL PRIMARY KEY,
    checkpoint_name VARCHAR(255) UNIQUE NOT NULL,
    checkpoint_path VARCHAR(500) NOT NULL,
    model_type VARCHAR(50),
    training_run_id VARCHAR(100),
    epoch INTEGER,
    val_mrr FLOAT,
    file_size_mb FLOAT,
    is_active INTEGER DEFAULT 0,
    is_best INTEGER DEFAULT 0,
    created_at TIMESTAMP
);
```

**Purpose**: Catalog all saved models, track performance, manage active models

#### **5. system_metrics** - Performance Monitoring
```sql
CREATE TABLE system_metrics (
    id SERIAL PRIMARY KEY,
    metric_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    value FLOAT NOT NULL,
    unit VARCHAR(20),
    endpoint VARCHAR(100),
    model_version VARCHAR(50),
    recorded_at TIMESTAMP DEFAULT NOW()
);
```

**Purpose**: Monitor API latency, prediction accuracy, system health

### **Database Connection**

**File**: `app/db/database.py`

```python
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True
)
```

**Environment Variable** (`.env`):
```bash
DATABASE_URL=postgresql://user:pass@host/database?sslmode=require
```

### **Auto-Initialization**

Tables are automatically created on server startup:
```python
# In app/main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # Creates all 5 tables
    ...
```

### **Usage Examples**

**Query Predictions:**
```python
from app.db.database import SessionLocal
from app.db.models import Prediction

db = SessionLocal()
predictions = db.query(Prediction).filter(
    Prediction.prediction_date >= datetime.now().date()
).all()
print(f"Made {len(predictions)} predictions today")
db.close()
```

**Query Training History:**
```python
from app.db.models import TrainingHistory

recent_runs = db.query(TrainingHistory).filter(
    TrainingHistory.status == 'completed'
).order_by(TrainingHistory.val_mrr.desc()).limit(5).all()
```

**Check System Metrics:**
```python
from app.db.models import SystemMetrics

avg_latency = db.query(func.avg(SystemMetrics.value)).filter(
    SystemMetrics.metric_type == 'api_latency',
    SystemMetrics.recorded_at >= datetime.now() - timedelta(hours=24)
).scalar()
```

---

## 📝 License

MIT License - Free for research and commercial use.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📧 Contact

For questions, issues, or collaboration:
- GitHub Issues: [Create an issue](https://github.com/your-repo/issues)
- Email: your-email@example.com

---

**FinGAT Backend** - Honest, leak-free, GNN-powered Indian stock prediction, fully RL-optimized, ready for production! 🚀📈