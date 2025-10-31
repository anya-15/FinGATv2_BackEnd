# FinGAT Backend API - Indian Stock Prediction

**Graph Attention Network (GATv2) for NSE/BSE Stock Market Predictions**

Automated daily training + FastAPI backend + PostgreSQL database

---

## 🚀 Features

- 📈 **Top-K Stock Predictions** - Predict most profitable Indian stocks (NSE/BSE)
- 🤖 **Daily Auto-Training** - Model retrains automatically at 6:30 PM IST after market close
- 🎯 **Movement Prediction** - Predict Up/Down with confidence scores
- 📊 **Sector Filtering** - Filter predictions by sector (Technology, Finance, etc.)
- ⚡ **Fast Inference** - Powered by PyTorch Geometric + Lightning
- 🗄️ **PostgreSQL Database** - Stock metadata stored in Neon (free tier)

---

## 📁 Project Structure

fingat-backend/
├── app/ # FastAPI application
│ ├── api/ # API routes
│ ├── core/ # Model loader & predictor
│ ├── db/ # Database models
│ ├── scheduler/ # Daily training automation
│ └── schemas/ # Response models
├── training/ # Your FinGAT model
│ └── lightning_module.py
├── data/ # Data loading logic
│ └── data_loader.py
├── indian_data/ # CSV files for NSE/BSE stocks
├── config/ # Configuration
│ └── config.yaml
├── scripts/ # Automation scripts
│ ├── train_model.py
│ ├── update_data.py
│ └── populate_db.py
├── checkpoints/ # Trained models
├── .env # Environment variables
└── requirements.txt # Dependencies

text

---

## 🛠️ Setup Instructions

### 1. Clone & Navigate
git clone <your-repo-url>
cd fingat-backend

text

### 2. Create Virtual Environment
python -m venv venv

Activate
Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

text

### 3. Install Dependencies
pip install -r requirements.txt

text

### 4. Setup PostgreSQL (Neon)
1. Go to https://neon.tech
2. Sign up (free, no credit card)
3. Create project: `fingat-db`
4. Select region: **Singapore** (closest to India)
5. Copy connection string

### 5. Configure Environment
Create `.env` file:
DATABASE_URL=postgresql://your_connection_string_from_neon
DATA_PATH=indian_data
DEVICE=cpu
API_PORT=8000
TRAINING_HOUR=18
TRAINING_MINUTE=30
TRAINING_TIMEZONE=Asia/Kolkata

text

### 6. Add Your Data
Copy your Indian stock CSV files to indian_data/
cp /path/to/your/stocks/*.csv indian_data/

text

### 7. Populate Database
python scripts/populate_db.py

text

### 8. Train Initial Model
python scripts/train_model.py

text
⏳ Takes 10-30 minutes depending on data size

### 9. Start API Server
uvicorn app.main:app --reload --port 8000

text

---

## 🌐 API Endpoints

### Health Check
GET http://localhost:8000/api/v1/health

text

### Get Top-K Predictions
GET http://localhost:8000/api/v1/predict/top-k?k=10&sector=Technology

text

**Response:**
{
"timestamp": "2025-10-28T11:30:00",
"k": 10,
"sector": "Technology",
"predictions": [
{
"rank": 1,
"ticker": "TCS",
"company_name": "Tata Consultancy Services",
"current_price": 3450.50,
"predicted_movement": "up",
"movement_percentage": 4.2,
"confidence_score": 0.89,
"sector": "Technology"
}
]
}

text

### List Sectors
GET http://localhost:8000/api/v1/sectors

text

### Manual Retrain
POST http://localhost:8000/api/v1/retrain

text

### API Documentation
http://localhost:8000/docs

text

---

## ⏰ Daily Automation

The system automatically:
1. **Updates data** from yfinance at 6:30 PM IST
2. **Retrains model** with latest data
3. **Hot-reloads** new model into API (no restart needed)

**Schedule:** Every day at 18:30 IST (after NSE/BSE market closes at 15:30)

---

## 📊 Model Architecture

- **Base:** Graph Attention Network v2 (GATv2)
- **Framework:** PyTorch Lightning + torch-geometric
- **Features:** 36 technical indicators per stock
- **Tasks:** Multi-task learning
  - Return prediction (regression)
  - Movement classification (up/down)
  - Stock ranking (listwise)
- **Stocks:** 550 NSE/BSE companies
- **Prediction Horizon:** 7 days

---

## 🧪 Testing

### Test Database Connection
python -c "from app.db.database import test_connection; test_connection()"

text

### Test Model Loading
python -c "from app.core.model_loader import model_loader; model_loader.load_model()"

text

### Manual Data Update
python scripts/update_data.py

text

---

## 🚀 Deployment

### Deploy to Render/Railway
1. Push to GitHub
2. Connect repository
3. Set environment variables
4. Deploy!

### Environment Variables for Production
DATABASE_URL=<your_neon_connection_string>
DEVICE=cpu
API_PORT=8000

text

---

## 📝 Notes

- **Data Format:** CSVs must have columns: `Date,Open,High,Low,Close,Volume,name`
- **Training Time:** ~10-30 minutes on CPU for 550 stocks
- **Database:** Free Neon PostgreSQL tier (512 MB storage)
- **Model Size:** ~50-200 MB checkpoint file

---

## 🐛 Troubleshooting

### Model not loading
Train a model first
python scripts/train_model.py

text

### Database connection failed
Check .env file has correct DATABASE_URL
Test connection
python -c "from app.db.database import test_connection; test_connection()"

text

### Daily training not running
Check logs in terminal where API is running
Manually trigger: POST /api/v1/retrain
text

---

## 📧 Support

Built with ❤️ for Indian stock market prediction using Graph Neural Networks

---

## 📄 License

MIT License - Free to use for educational/commercial purposes
How to create:

bash
nano README.md

# Copy-paste the content above
# Save: Ctrl+O, Enter, Ctrl+X
