"""
Database Connection and Session Management
Connects to PostgreSQL (Neon) using SQLAlchemy
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from app.config import settings

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=False,  # Set to True to see SQL queries in logs
    pool_pre_ping=True,  # Verify connections before using
    pool_size=5,
    max_overflow=10
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for models
Base = declarative_base()


def get_db() -> Session:
    """
    Dependency for FastAPI routes
    Provides database session and closes it after request
    
    Usage:
        @app.get("/stocks")
        def get_stocks(db: Session = Depends(get_db)):
            stocks = db.query(Stock).all()
            return stocks
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database tables
    Creates all tables defined in models.py
    
    Call this on app startup
    """
    from app.db.models import Stock  # Import models
    
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables initialized")
    except Exception as e:
        print(f"⚠️ Database initialization warning: {e}")


def test_connection():
    """
    Test database connection
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
