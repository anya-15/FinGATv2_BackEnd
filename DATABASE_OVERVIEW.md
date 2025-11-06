# 🗄️ FinGAT Database Overview

## Database Connection Details

### **Provider: Neon PostgreSQL**
- **Type**: Serverless PostgreSQL
- **Region**: ap-southeast-1 (Singapore)
- **SSL**: Required
- **Connection Pooling**: Enabled

### **Current Status**
```
✅ Connected: Yes
✅ Tables: 6 tables
✅ Stocks: 148 records
✅ Pool Size: 5 connections
✅ SSL Mode: Required
```

---

## 📊 Database Schema

### **1. stocks** (Primary Table)
Stores metadata about NSE/BSE stocks

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key (auto-increment) |
| `ticker` | String(20) | Stock ticker symbol (unique, indexed) |
| `company_name` | String(255) | Full company name |
| `sector` | String(100) | Business sector (indexed) |
| `industry` | String(100) | Industry classification |
| `current_price` | Float | Latest stock price |
| `market_cap` | Float | Market capitalization |
| `exchange` | String(10) | NSE or BSE |
| `description` | Text | Company description |
| `website` | String(255) | Company website |
| `created_at` | DateTime | Record creation timestamp |
| `updated_at` | DateTime | Last update timestamp |

**Sample Data:**
```
AMBUJACEM    - AMBUJACEM      (Construction)
APOLLOHOSP   - APOLLOHOSP     (Healthcare)
BAJAJ-AUTO   - BAJAJ-AUTO     (Automotive)
BAJAJFINSV   - BAJAJFINSV     (Finance)
BAJAJHLDNG   - BAJAJHLDNG     (Finance)
```

### **2. predictions** (Optional)
Stores historical prediction results

### **3. training_history** (Optional)
Tracks model training runs

### **4. rl_experiments** (Optional)
Logs RL optimization experiments

### **5. model_checkpoints** (Optional)
Metadata about saved model checkpoints

### **6. system_metrics** (Optional)
System performance and health metrics

---

## 🔌 Connection Configuration

### **File: `app/db/database.py`**

```python
# Database engine with connection pooling
engine = create_engine(
    settings.DATABASE_URL,
    echo=False,              # Set to True for SQL query logging
    pool_pre_ping=True,      # Verify connections before use
    pool_size=5,             # 5 persistent connections
    max_overflow=10          # Up to 10 additional connections
)

# Session factory for creating database sessions
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)
```

### **Environment Variable: `.env`**

```bash
DATABASE_URL=postgresql://username:password@host/database?sslmode=require
```

**Format:**
```
postgresql://[user]:[password]@[host]/[database]?sslmode=require
```

---

## 🔧 Database Functions

### **1. get_db()** - Session Dependency
```python
def get_db() -> Session:
    """
    FastAPI dependency for database sessions
    Automatically closes session after request
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**Usage in API:**
```python
@router.get("/stocks")
async def list_stocks(db: Session = Depends(get_db)):
    stocks = db.query(Stock).all()
    return stocks
```

### **2. init_db()** - Initialize Tables
```python
def init_db():
    """
    Creates all database tables
    Called on app startup
    """
    Base.metadata.create_all(bind=engine)
```

### **3. test_connection()** - Connection Test
```python
def test_connection():
    """
    Verify database connectivity
    Returns True if successful
    """
    db = SessionLocal()
    db.execute("SELECT 1")
    db.close()
```

---

## 📡 API Endpoints Using Database

### **GET /api/v1/stocks**
Lists all stocks from database
```python
@router.get("/stocks")
async def list_stocks(
    sector: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    query = db.query(Stock)
    if sector:
        query = query.filter(Stock.sector == sector)
    stocks = query.limit(limit).all()
    return [stock.to_dict() for stock in stocks]
```

### **GET /api/v1/sectors**
Lists unique sectors
```python
@router.get("/sectors")
async def get_sectors(db: Session = Depends(get_db)):
    sectors = db.query(Stock.sector).distinct().all()
    return {"sectors": [s[0] for s in sectors if s[0]]}
```

### **GET /api/v1/predict/top-k**
Predictions with database filtering
```python
@router.get("/predict/top-k")
async def get_top_k_stocks(
    k: int = 10,
    sector: Optional[str] = None,
    db: Session = Depends(get_db)
):
    # Get predictions from model
    predictions = predictor.predict_top_k(db=db, k=k, sector=sector)
    return predictions
```

---

## 🔄 Data Flow

### **1. Data Update Flow**
```
CSV Files (indian_data/)
    ↓
update_data.py (fetch latest prices)
    ↓
populate_db.py (sync to database)
    ↓
PostgreSQL Database (stocks table)
    ↓
API Endpoints (query stocks)
```

### **2. Prediction Flow**
```
API Request
    ↓
Database Query (get stock metadata)
    ↓
CSV Files (get historical prices)
    ↓
Model Prediction
    ↓
Combine Results (metadata + predictions)
    ↓
API Response
```

---

## 🛠️ Database Operations

### **Populate Database**
```bash
python scripts/populate_db.py
```
- Reads CSV files from `indian_data/`
- Extracts stock metadata
- Inserts/updates records in database
- Maps sectors from `config/sector_mapping.json`

### **Query Stocks**
```python
from app.db.database import SessionLocal
from app.db.models import Stock

db = SessionLocal()

# Get all stocks
stocks = db.query(Stock).all()

# Filter by sector
finance_stocks = db.query(Stock).filter(Stock.sector == "Finance").all()

# Get specific stock
reliance = db.query(Stock).filter(Stock.ticker == "RELIANCE").first()

db.close()
```

### **Update Stock**
```python
db = SessionLocal()

stock = db.query(Stock).filter(Stock.ticker == "RELIANCE").first()
stock.current_price = 2500.50
stock.market_cap = 1700000000000

db.commit()
db.close()
```

---

## 📊 Current Database State

**Connection:**
```
Host: ep-mute-cherry-a1g8kkbq-pooler.ap-southeast-1.aws.neon.tech
Database: neondb
Driver: postgresql
SSL: Required
```

**Tables:**
```
✅ stocks              (148 records)
✅ predictions         (optional)
✅ training_history    (optional)
✅ rl_experiments      (optional)
✅ model_checkpoints   (optional)
✅ system_metrics      (optional)
```

**Connection Pool:**
```
Pool Size: 5 connections
Max Overflow: 10 connections
Pre-ping: Enabled (validates before use)
```

---

## 🔒 Security

### **SSL/TLS**
- All connections use SSL (`sslmode=require`)
- Encrypted data transmission

### **Credentials**
- Stored in `.env` file (gitignored)
- Never committed to repository
- Use `.env.example` as template

### **Connection Pooling**
- Reuses connections efficiently
- Prevents connection exhaustion
- Auto-closes idle connections

---

## 🧪 Testing Database

### **Test Connection**
```bash
python test_db_connection.py
```

**Output:**
```
✅ Connection successful
✅ Found 6 tables
✅ 148 stocks in database
✅ Pool size: 5
```

### **Manual Test**
```python
from app.db.database import test_connection
test_connection()  # Returns True if successful
```

---

## 📝 Database vs CSV Files

| Aspect | Database (PostgreSQL) | CSV Files |
|--------|----------------------|-----------|
| **Purpose** | Stock metadata | Historical OHLC data |
| **Data** | Company info, sectors | Daily prices, volume |
| **Updates** | Via populate_db.py | Via update_data.py |
| **Used By** | API endpoints | Model training/prediction |
| **Size** | 148 records | 147 files × 1250 rows |
| **Query Speed** | Fast (indexed) | Slower (file I/O) |

**Both are used together:**
- Database: Quick metadata lookups
- CSV Files: Historical price analysis

---

## 🚀 Production Considerations

### **Scaling**
- Neon auto-scales based on load
- Connection pooling handles concurrent requests
- Consider read replicas for heavy traffic

### **Backup**
- Neon provides automatic backups
- Point-in-time recovery available
- Export data periodically for safety

### **Monitoring**
- Check connection pool usage
- Monitor query performance
- Track database size growth

### **Optimization**
- Indexes on `ticker` and `sector` columns
- Connection pooling reduces overhead
- Pre-ping validates connections

---

## 📚 Related Files

- `app/db/database.py` - Connection setup
- `app/db/models.py` - Table schemas
- `scripts/populate_db.py` - Data population
- `test_db_connection.py` - Connection testing
- `.env` - Database credentials (gitignored)
- `.env.example` - Configuration template

---

**Last Updated:** November 6, 2025  
**Status:** ✅ Connected and Operational  
**Records:** 148 stocks across 22 sectors
