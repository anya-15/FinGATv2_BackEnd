"""
Scheduler Tasks - Daily Training Pipeline
Automatically updates data and retrains model after market close
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.config import settings
from app.core.model_loader import model_loader


def update_stock_data():
    """
    Step 1: Update stock data and sync database
    Runs: scripts/sync_data.py (combines update + populate)
    """
    print("=" * 60)
    print(f"[*] Starting data sync - {datetime.now()}")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, "scripts/sync_data.py"],
            capture_output=True,
            text=True,
            timeout=900  # 15 minutes timeout (includes DB sync)
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print("[OK] Data sync completed successfully")
            return True
        else:
            print(f"[WARNING] Data sync failed with code {result.returncode}")
            print(result.stderr)
            return False
    
    except subprocess.TimeoutExpired:
        print("[ERROR] Data sync timed out (>15 minutes)")
        return False
    except Exception as e:
        print(f"[ERROR] Data sync error: {e}")
        return False


def train_model():
    """
    Step 2: Train FinGAT model with updated data using Hybrid RL
    Runs: scripts/train_with_hybrid_rl.py
    """
    print("=" * 60)
    print(f"[*] Starting Hybrid RL model training - {datetime.now()}")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, "scripts/train_with_hybrid_rl.py"],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout for RL training
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print("[OK] Model training completed successfully")
            return True
        else:
            print(f"[WARNING] Model training failed with code {result.returncode}")
            print(result.stderr)
            return False
    
    except subprocess.TimeoutExpired:
        print("[ERROR] Model training timed out (>1 hour)")
        return False
    except Exception as e:
        print(f"[ERROR] Model training error: {e}")
        return False


def reload_model():
    """
    Step 3: Reload new model into API
    Hot-reload without restarting server
    """
    print("=" * 60)
    print(f"[*] Reloading model - {datetime.now()}")
    print("=" * 60)
    
    try:
        model_loader.reload_model()
        print("[OK] Model reloaded successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Model reload error: {e}")
        return False


def daily_pipeline():
    """
    Complete daily pipeline:
    1. Sync data (Update CSVs + Sync Database)
    2. Retrain FinGAT model with Hybrid RL
    3. Reload model into API
    
    Scheduled to run at 6:30 PM IST (after NSE/BSE market closes at 3:30 PM)
    """
    print("\n\n")
    print("=" * 60)
    print("[*] DAILY FINGAT PIPELINE STARTED")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # Step 1: Update data
    data_success = update_stock_data()
    
    if not data_success:
        print("\n[ERROR] Pipeline failed at data update stage")
        return
    
    # Step 2: Train model
    train_success = train_model()
    
    if not train_success:
        print("\n[ERROR] Pipeline failed at training stage")
        return
    
    # Step 3: Reload model
    reload_success = reload_model()
    
    if not reload_success:
        print("\n[WARNING] Model trained but reload failed")
        print("   Restart API to load new model")
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print("[OK] DAILY PIPELINE COMPLETED")
    print(f"   Duration: {duration:.0f} seconds ({duration/60:.1f} minutes)")
    print(f"   Data Update: {'[OK]' if data_success else '[ERROR]'}")
    print(f"   Model Training: {'[OK]' if train_success else '[ERROR]'}")
    print(f"   Model Reload: {'[OK]' if reload_success else '[ERROR]'}")
    print("=" * 60)
    print("\n\n")


def setup_scheduler():
    """
    Setup APScheduler for daily training
    
    Returns:
        BackgroundScheduler instance
    """
    scheduler = BackgroundScheduler(timezone=settings.TRAINING_TIMEZONE)
    
    # Schedule daily training at 6:30 PM IST
    scheduler.add_job(
        daily_pipeline,
        trigger=CronTrigger(
            hour=settings.TRAINING_HOUR,
            minute=settings.TRAINING_MINUTE,
            timezone=settings.TRAINING_TIMEZONE
        ),
        id='daily_training_pipeline',
        name='FinGAT Daily Training Pipeline',
        replace_existing=True,
        misfire_grace_time=300  # 5 minutes grace period
    )
    
    print(f"[*] Scheduled daily training at {settings.TRAINING_HOUR}:{settings.TRAINING_MINUTE:02d} {settings.TRAINING_TIMEZONE}")
    
    return scheduler


# For manual testing
if __name__ == "__main__":
    print("Running daily pipeline manually...")
    daily_pipeline()
