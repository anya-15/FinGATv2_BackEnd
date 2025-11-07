"""
Update Indian Stock Data (NSE/BSE)
Downloads latest stock prices and appends to CSV files
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys
import time
import warnings

# Suppress yfinance warnings
warnings.filterwarnings('ignore')
yf.set_tz_cache_location(None)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import settings


def update_indian_stocks():
    """
    Update all NSE/BSE stock CSV files with latest data
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    print("=" * 60)
    print(f"📡 Updating Indian Stock Data (NSE/BSE)")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    data_dir = Path(settings.DATA_PATH)
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    # Get list of existing stocks
    stock_files = list(data_dir.glob("*.csv"))
    
    if len(stock_files) == 0:
        print(f"❌ No stock files found in {data_dir}/")
        return False
    
    print(f"📊 Found {len(stock_files)} Indian stocks to update\n")
    
    updated_count = 0
    failed_count = 0
    no_data_count = 0
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
            
            # NSE stocks have .NS suffix, BSE have .BO
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
                
                # Add 'name' column if exists in original
                if 'name' in df.columns:
                    company_name = df['name'].iloc[0]
                    new_data['name'] = company_name
                
                # Append new data
                updated_df = pd.concat([df, new_data], ignore_index=True)
                updated_df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
                updated_df.to_csv(csv_file, index=False)
                
                print(f"[{idx}/{total_files}] ✅ {ticker:15s} ({success_symbol:15s}) +{len(new_data)} rows")
                updated_count += 1
            else:
                print(f"[{idx}/{total_files}] ⏭️  {ticker:15s} - No new data")
                no_data_count += 1
        
        except Exception as e:
            print(f"[{idx}/{total_files}] ❌ {ticker:15s} - Error: {str(e)[:30]}")
            failed_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Update Summary:")
    print(f"   Updated: {updated_count}")
    print(f"   No new data: {no_data_count}")
    print(f"   Failed: {failed_count}")
    print("=" * 60)
    
    return updated_count > 0 or no_data_count > 0


if __name__ == '__main__':
    success = update_indian_stocks()
    sys.exit(0 if success else 1)
