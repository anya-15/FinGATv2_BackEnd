"""
Database Models - Table Schemas
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import func
from app.db.database import Base


class Stock(Base):
    """
    Stock table - stores metadata about NSE/BSE stocks
    
    This complements your CSV files which have historical price data.
    Database stores current metadata like company name, sector, etc.
    """
    
    __tablename__ = "stocks"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Stock identification
    ticker = Column(String(20), unique=True, nullable=False, index=True)
    company_name = Column(String(255), nullable=False)
    
    # Classification
    sector = Column(String(100), nullable=True, index=True)
    industry = Column(String(100), nullable=True)
    
    # Market data
    current_price = Column(Float, nullable=True)
    market_cap = Column(Float, nullable=True)
    
    # Exchange info
    exchange = Column(String(10), default="NSE")  # NSE or BSE
    
    # Additional info
    description = Column(Text, nullable=True)
    website = Column(String(255), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<Stock(ticker={self.ticker}, company={self.company_name}, sector={self.sector})>"
    
    def to_dict(self):
        """Convert to dictionary for JSON responses"""
        return {
            "id": self.id,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "sector": self.sector,
            "industry": self.industry,
            "current_price": float(self.current_price) if self.current_price else None,
            "market_cap": float(self.market_cap) if self.market_cap else None,
            "exchange": self.exchange,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
