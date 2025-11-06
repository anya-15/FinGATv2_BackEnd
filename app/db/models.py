"""
Database Models - Table Schemas
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
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


class Prediction(Base):
    """
    Prediction table - stores historical stock predictions
    
    Tracks all predictions made by the model for analysis and backtesting.
    """
    
    __tablename__ = "predictions"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Stock reference
    ticker = Column(String(20), nullable=False, index=True)
    
    # Prediction details
    predicted_return = Column(Float, nullable=False)
    predicted_movement = Column(String(10), nullable=False)  # 'up' or 'down'
    movement_confidence = Column(Float, nullable=False)  # 0.0 to 1.0
    ranking_score = Column(Float, nullable=False)
    rank = Column(Integer, nullable=True)  # Rank in top-K
    
    # Model info
    model_checkpoint = Column(String(255), nullable=True)
    model_version = Column(String(50), nullable=True)
    
    # Metadata
    sector = Column(String(100), nullable=True)
    prediction_date = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Actual outcome (to be filled later for backtesting)
    actual_return = Column(Float, nullable=True)
    actual_movement = Column(String(10), nullable=True)
    outcome_date = Column(DateTime(timezone=True), nullable=True)
    
    # Accuracy tracking
    is_correct = Column(Integer, nullable=True)  # 1 for correct, 0 for incorrect, NULL for pending
    
    def __repr__(self):
        return f"<Prediction(ticker={self.ticker}, movement={self.predicted_movement}, date={self.prediction_date})>"
    
    def to_dict(self):
        """Convert to dictionary for JSON responses"""
        return {
            "id": self.id,
            "ticker": self.ticker,
            "predicted_return": float(self.predicted_return) if self.predicted_return else None,
            "predicted_movement": self.predicted_movement,
            "movement_confidence": float(self.movement_confidence) if self.movement_confidence else None,
            "ranking_score": float(self.ranking_score) if self.ranking_score else None,
            "rank": self.rank,
            "model_checkpoint": self.model_checkpoint,
            "model_version": self.model_version,
            "sector": self.sector,
            "prediction_date": self.prediction_date.isoformat() if self.prediction_date else None,
            "actual_return": float(self.actual_return) if self.actual_return else None,
            "actual_movement": self.actual_movement,
            "outcome_date": self.outcome_date.isoformat() if self.outcome_date else None,
            "is_correct": self.is_correct
        }


class TrainingHistory(Base):
    """
    Training History table - tracks all model training runs
    
    Stores metrics and configuration for each training session.
    """
    
    __tablename__ = "training_history"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Training session info
    run_id = Column(String(100), unique=True, nullable=False, index=True)
    training_type = Column(String(50), nullable=False)  # 'classical', 'rl_hybrid', 'rl_features', etc.
    
    # Model configuration
    model_architecture = Column(String(50), nullable=True)  # 'GATv2', 'GraphSAGE', etc.
    hidden_dim = Column(Integer, nullable=True)
    num_layers = Column(Integer, nullable=True)
    num_heads = Column(Integer, nullable=True)
    dropout = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=True)
    
    # Training metrics
    num_epochs = Column(Integer, nullable=True)
    best_epoch = Column(Integer, nullable=True)
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    test_loss = Column(Float, nullable=True)
    
    # Performance metrics
    val_accuracy = Column(Float, nullable=True)
    val_mrr = Column(Float, nullable=True)
    val_precision_at_10 = Column(Float, nullable=True)
    test_accuracy = Column(Float, nullable=True)
    test_mrr = Column(Float, nullable=True)
    
    # Feature selection (for RL)
    num_features_selected = Column(Integer, nullable=True)
    total_features = Column(Integer, nullable=True)
    feature_mask_path = Column(String(500), nullable=True)
    
    # Checkpoint info
    checkpoint_path = Column(String(500), nullable=True)
    checkpoint_size_mb = Column(Float, nullable=True)
    
    # Timing
    training_duration_minutes = Column(Float, nullable=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Status
    status = Column(String(20), nullable=False, default='running')  # 'running', 'completed', 'failed'
    error_message = Column(Text, nullable=True)
    
    # Additional metadata
    notes = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<TrainingHistory(run_id={self.run_id}, type={self.training_type}, status={self.status})>"
    
    def to_dict(self):
        """Convert to dictionary for JSON responses"""
        return {
            "id": self.id,
            "run_id": self.run_id,
            "training_type": self.training_type,
            "model_architecture": self.model_architecture,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": float(self.dropout) if self.dropout else None,
            "learning_rate": float(self.learning_rate) if self.learning_rate else None,
            "num_epochs": self.num_epochs,
            "best_epoch": self.best_epoch,
            "train_loss": float(self.train_loss) if self.train_loss else None,
            "val_loss": float(self.val_loss) if self.val_loss else None,
            "test_loss": float(self.test_loss) if self.test_loss else None,
            "val_accuracy": float(self.val_accuracy) if self.val_accuracy else None,
            "val_mrr": float(self.val_mrr) if self.val_mrr else None,
            "val_precision_at_10": float(self.val_precision_at_10) if self.val_precision_at_10 else None,
            "test_accuracy": float(self.test_accuracy) if self.test_accuracy else None,
            "test_mrr": float(self.test_mrr) if self.test_mrr else None,
            "num_features_selected": self.num_features_selected,
            "total_features": self.total_features,
            "feature_mask_path": self.feature_mask_path,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_size_mb": float(self.checkpoint_size_mb) if self.checkpoint_size_mb else None,
            "training_duration_minutes": float(self.training_duration_minutes) if self.training_duration_minutes else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "error_message": self.error_message,
            "notes": self.notes
        }


class ModelCheckpoint(Base):
    """
    Model Checkpoints table - catalog of all saved model checkpoints
    
    Tracks metadata about each checkpoint for easy management.
    """
    
    __tablename__ = "model_checkpoints"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Checkpoint identification
    checkpoint_name = Column(String(255), unique=True, nullable=False, index=True)
    checkpoint_path = Column(String(500), nullable=False)
    
    # Model info
    model_type = Column(String(50), nullable=True)  # 'FinGAT', 'GATv2', etc.
    model_version = Column(String(50), nullable=True)
    
    # Training info
    training_run_id = Column(String(100), nullable=True, index=True)
    epoch = Column(Integer, nullable=True)
    
    # Performance metrics
    val_loss = Column(Float, nullable=True)
    val_accuracy = Column(Float, nullable=True)
    val_mrr = Column(Float, nullable=True)
    test_accuracy = Column(Float, nullable=True)
    
    # File info
    file_size_mb = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Status
    is_active = Column(Integer, default=0)  # 1 if currently in use, 0 otherwise
    is_best = Column(Integer, default=0)  # 1 if best model, 0 otherwise
    
    # Metadata
    description = Column(Text, nullable=True)
    tags = Column(String(500), nullable=True)  # Comma-separated tags
    
    def __repr__(self):
        return f"<ModelCheckpoint(name={self.checkpoint_name}, val_mrr={self.val_mrr})>"
    
    def to_dict(self):
        """Convert to dictionary for JSON responses"""
        return {
            "id": self.id,
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_path": self.checkpoint_path,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "training_run_id": self.training_run_id,
            "epoch": self.epoch,
            "val_loss": float(self.val_loss) if self.val_loss else None,
            "val_accuracy": float(self.val_accuracy) if self.val_accuracy else None,
            "val_mrr": float(self.val_mrr) if self.val_mrr else None,
            "test_accuracy": float(self.test_accuracy) if self.test_accuracy else None,
            "file_size_mb": float(self.file_size_mb) if self.file_size_mb else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_active": bool(self.is_active),
            "is_best": bool(self.is_best),
            "description": self.description,
            "tags": self.tags.split(',') if self.tags else []
        }


class SystemMetrics(Base):
    """
    System Metrics table - tracks system performance and health
    
    Monitors API performance, prediction accuracy, and system health over time.
    """
    
    __tablename__ = "system_metrics"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Metric identification
    metric_type = Column(String(50), nullable=False, index=True)  # 'api_latency', 'prediction_accuracy', 'memory_usage', etc.
    metric_name = Column(String(100), nullable=False, index=True)
    
    # Metric value
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=True)  # 'ms', 'MB', '%', etc.
    
    # Context
    endpoint = Column(String(100), nullable=True)  # For API metrics
    model_version = Column(String(50), nullable=True)
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Additional data
    extra_data = Column(Text, nullable=True)  # JSON string for additional context
    
    def __repr__(self):
        return f"<SystemMetrics(type={self.metric_type}, name={self.metric_name}, value={self.value})>"
    
    def to_dict(self):
        """Convert to dictionary for JSON responses"""
        return {
            "id": self.id,
            "metric_type": self.metric_type,
            "metric_name": self.metric_name,
            "value": float(self.value),
            "unit": self.unit,
            "endpoint": self.endpoint,
            "model_version": self.model_version,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
            "extra_data": self.extra_data
        }
