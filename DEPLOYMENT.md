# FinGAT Backend - Deployment Guide

## Local Development

1. Activate virtual environment
source venv/bin/activate # Linux/Mac
venv\Scripts\activate.bat # Windows

2. Create .env file with database credentials
3. Populate database
python scripts/populate_db.py

4. Train model
python scripts/train_model.py

5. Run server
uvicorn app.main:app --reload --port 8000

text

## Production Deployment

### Render (Recommended)

1. Push code to GitHub
2. Go to https://render.com
3. Create new Web Service
4. Connect GitHub repository
5. Set build command: `pip install -r requirements.txt`
6. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
7. Add environment variables in dashboard:
   - `DATABASE_URL`
   - `DEVICE=cpu`

### Railway

npm install -g @railway/cli
railway login
railway init
railway up

text

## Daily Automation

- Model automatically retrains at 6:30 PM IST
- Data updates from yfinance
- New model checkpoint saved
- API hot-reloads without restart

## API Endpoints

- `GET /api/v1/health` - Health check
- `GET /api/v1/predict/top-k` - Top K predictions
- `GET /api/v1/sectors` - List sectors
- `GET /api/v1/stocks` - List all stocks
- `GET /api/v1/model/status` - Model training status
- `POST /api/v1/retrain` - Manual retrain

## Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
✅ STEP 8: Final Testing
Run all endpoints once more:

bash
# Test in new terminal
# All should return 200 OK

curl -X GET http://localhost:8000/api/v1/health
curl -X GET "http://localhost:8000/api/v1/predict/top-k?k=10"
curl -X GET "http://localhost:8000/api/v1/predict/top-k?k=5&sector=Technology"
curl -X GET http://localhost:8000/api/v1/sectors
curl -X GET http://localhost:8000/api/v1/stocks?limit=5
curl -X GET http://localhost:8000/api/v1/model/status
curl -X GET http://localhost:8000/api/v1/model/info
✅ STEP 9: Version Control
bash
# Initialize git (if not done)
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit - FinGAT Backend v1.0"

# Add remote (if using GitHub)
git remote add origin https://github.com/yourusername/fingat-backend.git

# Push
git push -u origin main
📋 Final Checklist - Backend Complete
✅ API running on http://localhost:8000

✅ All 7+ endpoints working

✅ Database connected (Neon PostgreSQL)

✅ Model trained and loaded

✅ Daily automation scheduled (6:30 PM IST)

✅ Price tracking from CSVs

✅ Sector filtering working

✅ Confidence scores generated

✅ .env configured

✅ .gitignore created

✅ requirements.txt complete

✅ Code in Git repository

✅ Ready for production deployment

🎉 Your Backend is COMPLETE!
You have built:

✅ Production-ready FastAPI backend

✅ PostgreSQL database integration

✅ Trained FinGAT model (GATv2)

✅ 148 Indian stocks analysis

✅ Daily automated retraining

✅ Real predictions with confidence

✅ Sector-based filtering

✅ Complete REST API

✅ Interactive API documentation

📦 Ready to Give Frontend Team
Share these endpoints with your frontend team:

javascript
// Base URL
const API_URL = "http://localhost:8000/api/v1";

// Or production URL
const API_URL = "https://your-production-url.com/api/v1";

// Endpoints:
GET  /health                          // Health check
GET  /predict/top-k?k=10              // Top 10 stocks
GET  /predict/top-k?k=10&sector=Tech  // Filter by sector
GET  /sectors                         // List sectors
GET  /stocks?limit=100                // List stocks
GET  /model/status                    // Model info
POST /retrain                         // Manual retrain