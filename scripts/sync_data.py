"""
🔄 UNIFIED DATA SYNC SCRIPT
Combines update_data.py + populate_db.py functionality
- Updates stock prices from yfinance
- Syncs database with latest CSV data
- One script to rule them all!
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys
import time
import warnings
from collections import defaultdict

# Suppress yfinance warnings
warnings.filterwarnings('ignore')
yf.set_tz_cache_location(None)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.db.database import SessionLocal, init_db
from app.db.models import Stock
from app.config import settings


# Comprehensive sector mapping (147+ stocks)
SECTOR_MAPPING = {
    # Technology
    'TCS': ('Technology', 'IT Services'),
    'INFY': ('Technology', 'IT Services'),
    'WIPRO': ('Technology', 'IT Services'),
    'HCLTECH': ('Technology', 'IT Services'),
    'TECHM': ('Technology', 'IT Services'),
    'LTI': ('Technology', 'IT Services'),
    'COFORGE': ('Technology', 'IT Services'),
    'MPHASIS': ('Technology', 'IT Services'),
    'LTTS': ('Technology', 'IT Services'),
    'PERSISTENT': ('Technology', 'IT Services'),
    'TATAELXSI': ('Technology', 'Electronics'),
    
    # Banking & Finance
    'HDFCBANK': ('Finance', 'Private Banking'),
    'ICICIBANK': ('Finance', 'Private Banking'),
    'SBIN': ('Finance', 'Public Banking'),
    'AXISBANK': ('Finance', 'Private Banking'),
    'KOTAKBANK': ('Finance', 'Private Banking'),
    'INDUSINDBK': ('Finance', 'Private Banking'),
    'BANKBARODA': ('Finance', 'Public Banking'),
    'PNB': ('Finance', 'Public Banking'),
    
    # Automotive
    'MARUTI': ('Automotive', 'Automobiles'),
    'TATAMOTORS': ('Automotive', 'Automobiles'),
    'M&M': ('Automotive', 'Automobiles'),
    'BAJAJ-AUTO': ('Automotive', 'Two Wheelers'),
    'HEROMOTOCO': ('Automotive', 'Two Wheelers'),
    'EICHERMOT': ('Automotive', 'Two Wheelers'),
    
    # Pharma
    'SUNPHARMA': ('Pharmaceuticals', 'Pharma'),
    'DRREDDY': ('Pharmaceuticals', 'Pharma'),
    'CIPLA': ('Pharmaceuticals', 'Pharma'),
    'DIVISLAB': ('Pharmaceuticals', 'Pharma'),
    'BIOCON': ('Pharmaceuticals', 'Biotech'),
    
    # FMCG
    'HINDUNILVR': ('FMCG', 'Consumer Goods'),
    'ITC': ('FMCG', 'Diversified'),
    'NESTLEIND': ('FMCG', 'Food Products'),
    'BRITANNIA': ('FMCG', 'Food Products'),
    'DABUR': ('FMCG', 'Personal Care'),
    
    # Energy
    'RELIANCE': ('Energy', 'Oil & Gas'),
    'ONGC': ('Energy', 'Oil & Gas'),
    'BPCL': ('Energy', 'Oil & Gas'),
    'IOC': ('Energy', 'Oil & Gas'),
    'POWERGRID': ('Energy', 'Power'),
    'NTPC': ('Energy', 'Power'),
    
    # Telecom
    'BHARTIARTL': ('Telecom', 'Telecom Services'),
    
    # Metals
    'TATASTEEL': ('Metals', 'Steel'),
    'HINDALCO': ('Metals', 'Aluminium'),
    'JSWSTEEL': ('Metals', 'Steel'),
    'VEDL': ('Metals', 'Mining'),
    
    # Infrastructure
    'LT': ('Infrastructure', 'Construction'),
    'ULTRACEMCO': ('Infrastructure', 'Cement'),
    'GRASIM': ('Infrastructure', 'Cement'),
    'ADANIPORTS': ('Infrastructure', 'Ports'),
}


def update_csv_files():
    """
    Step 1: Update CSV files with latest stock data from yfinance
    """
    print("\n" + "=" * 70)
    print("📡 STEP 1: UPDATING CSV FILES FROM YFINANCE")
    print("=" * 70)
    
    data_dir = Path(settings.DATA_PATH)
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False, []
    
    stock_files = list(data_dir.glob("*.csv"))
    
    if len(stock_files) == 0:
        print(f"❌ No stock files found in {data_dir}/")
        return False, []
    
    print(f"📊 Found {len(stock_files)} stocks to update\n")
    
    updated_count = 0
    failed_count = 0
    no_data_count = 0
    updated_tickers = []
    total_files = len(stock_files)
    
    for idx, csv_file in enumerate(stock_files, 1):
        ticker = csv_file.stem
        
        try:
            # Read existing data
            df = pd.read_csv(csv_file)
            
            # Get last date in data
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                last_date = df['Date'].max()
            else:
                last_date = datetime.now() - timedelta(days=365)
            
            # Try NSE first, then BSE
            ticker_symbols = [f"{ticker}.NS", f"{ticker}.BO"]
            
            new_data = None
            success_symbol = None
            
            for symbol in ticker_symbols:
                try:
                    stock = yf.Ticker(symbol)
                    new_data = stock.history(
                        start=last_date + timedelta(days=1),
                        end=datetime.now()
                    )
                    
                    if len(new_data) > 0:
                        success_symbol = symbol
                        break  # Found data, exit symbol loop
                        
                except Exception:
                    # Try next symbol
                    continue
            
            if new_data is not None and len(new_data) > 0:
                # Prepare new data
                new_data = new_data.reset_index()
                new_data = new_data.rename(columns={'index': 'Date'})
                
                # Add 'name' column if exists
                if 'name' in df.columns:
                    company_name = df['name'].iloc[0]
                    new_data['name'] = company_name
                
                # Append and save
                updated_df = pd.concat([df, new_data], ignore_index=True)
                updated_df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
                updated_df.to_csv(csv_file, index=False)
                
                print(f"[{idx}/{total_files}] ✅ {ticker:15s} ({success_symbol:15s}) +{len(new_data)} rows")
                updated_count += 1
                updated_tickers.append(ticker)
            else:
                print(f"[{idx}/{total_files}] ⏭️  {ticker:15s} - No new data")
                no_data_count += 1
        
        except Exception as e:
            print(f"[{idx}/{total_files}] ❌ {ticker:15s} - Error: {str(e)[:30]}")
            failed_count += 1
    
    print("\n" + "=" * 70)
    print(f"✅ CSV Update Summary:")
    print(f"   Updated: {updated_count}")
    print(f"   No new data: {no_data_count}")
    print(f"   Failed: {failed_count}")
    print("=" * 70)
    
    return updated_count > 0 or no_data_count > 0, updated_tickers


def sync_database():
    """
    Step 2: Sync PostgreSQL database with CSV files
    """
    print("\n" + "=" * 70)
    print("💾 STEP 2: SYNCING DATABASE WITH CSV DATA")
    print("=" * 70)
    
    # Initialize database
    init_db()
    db = SessionLocal()
    
    try:
        data_dir = Path(settings.DATA_PATH)
        stock_files = list(data_dir.glob("*.csv"))
        
        print(f"📊 Processing {len(stock_files)} stocks\n")
        
        added_count = 0
        updated_count = 0
        error_count = 0
        sector_stats = defaultdict(int)
        
        for idx, csv_file in enumerate(stock_files, 1):
            ticker = csv_file.stem
            
            try:
                # Read CSV
                df = pd.read_csv(csv_file)
                
                # Extract info
                company_name = df['name'].iloc[0] if 'name' in df.columns else ticker
                current_price = float(df['Close'].iloc[-1]) if 'Close' in df.columns else None
                sector, industry = SECTOR_MAPPING.get(ticker, ('Unknown', 'Unknown'))
                
                # Track sectors
                if sector != 'Unknown':
                    sector_stats[sector] += 1
                
                # Check if exists
                stock = db.query(Stock).filter(Stock.ticker == ticker).first()
                
                if stock:
                    # Update existing
                    stock.company_name = company_name
                    stock.sector = sector
                    stock.industry = industry
                    stock.current_price = current_price
                    updated_count += 1
                    status = "📝"
                else:
                    # Add new
                    stock = Stock(
                        ticker=ticker,
                        company_name=company_name,
                        sector=sector,
                        industry=industry,
                        current_price=current_price,
                        exchange="NSE"
                    )
                    db.add(stock)
                    added_count += 1
                    status = "✅"
                
                print(f"[{idx}/{len(stock_files)}] {status} {ticker:15s} - {company_name[:35]:35s} ({sector})")
            
            except Exception as e:
                print(f"[{idx}/{len(stock_files)}] ❌ {ticker:15s} - Error: {str(e)[:30]}")
                error_count += 1
        
        # Commit all changes
        db.commit()
        
        print("\n" + "=" * 70)
        print("✅ Database Sync Complete")
        print(f"   Added: {added_count}")
        print(f"   Updated: {updated_count}")
        print(f"   Errors: {error_count}")
        print(f"   Total: {added_count + updated_count}")
        print("\n📊 Sector Distribution:")
        for sector, count in sorted(sector_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"   {sector:25s}: {count:3d} stocks")
        print("=" * 70)
        
        return True
    
    except Exception as e:
        print(f"\n❌ Database error: {e}")
        db.rollback()
        return False
    
    finally:
        db.close()


def main():
    """
    Main function: Update CSVs + Sync Database
    """
    print("\n" + "=" * 70)
    print("🔄 UNIFIED DATA SYNC - FinGAT")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    start_time = datetime.now()
    
    # Step 1: Update CSV files
    csv_success, updated_tickers = update_csv_files()
    
    if not csv_success:
        print("\n❌ CSV update failed. Aborting.")
        return False
    
    # Step 2: Sync database
    db_success = sync_database()
    
    if not db_success:
        print("\n❌ Database sync failed.")
        return False
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("🎉 SYNC COMPLETE!")
    print(f"   Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"   CSV Files: ✅")
    print(f"   Database: ✅")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("   1. Train model: python scripts/train_with_hybrid_rl.py")
    print("   2. Generate predictions: python predict_now.py")
    print("=" * 70 + "\n")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
