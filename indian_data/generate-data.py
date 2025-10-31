import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time

# ✅ Indian NSE stocks
tickers = [
    'RELIANCE.NS','TCS.NS','HDFCBANK.NS','ICICIBANK.NS','INFY.NS','LT.NS','SBIN.NS','KOTAKBANK.NS','AXISBANK.NS','ITC.NS',
    'BHARTIARTL.NS','HCLTECH.NS','BAJFINANCE.NS','HINDUNILVR.NS','MARUTI.NS','ASIANPAINT.NS','SUNPHARMA.NS','ADANIGREEN.NS','ADANIPORTS.NS','ADANIPOWER.NS',
    'ADANIENT.NS','GRASIM.NS','ULTRACEMCO.NS','SHREECEM.NS','JSWSTEEL.NS','TATASTEEL.NS','COALINDIA.NS','ONGC.NS','NTPC.NS','POWERGRID.NS',
    'BPCL.NS','IOC.NS','HDFCLIFE.NS','SBILIFE.NS','ICICIPRULI.NS','BAJAJFINSV.NS','BAJAJ-AUTO.NS','M&M.NS','TATAMOTORS.NS','EICHERMOT.NS',
    'DIVISLAB.NS','DRREDDY.NS','CIPLA.NS','APOLLOHOSP.NS','BRITANNIA.NS','NESTLEIND.NS','TITAN.NS','HEROMOTOCO.NS','BAJAJHLDNG.NS','UPL.NS',
    'GAIL.NS','HINDALCO.NS','VEDL.NS','TECHM.NS','WIPRO.NS','DLF.NS','SRF.NS','PIDILITIND.NS','DMART.NS','JUBLFOOD.NS',
    'INDUSINDBK.NS','BANKBARODA.NS','PNB.NS','CANBK.NS','IDFCFIRSTB.NS','BANDHANBNK.NS','AUBANK.NS','UCOBANK.NS','FEDERALBNK.NS','UNIONBANK.NS',
    'YESBANK.NS','IDBI.NS','LICHSGFIN.NS','HDFCAMC.NS','ICICIGI.NS','SBICARD.NS','MUTHOOTFIN.NS',
    'CHOLAFIN.NS','SUNDARMFIN.NS','RECLTD.NS','PFC.NS','PEL.NS','MOTILALOFS.NS',
    'ABB.NS','ABBOTINDIA.NS','ACC.NS','ADVENZYMES.NS','ALKEM.NS','AMBUJACEM.NS','AARTIIND.NS',
    'ASHOKLEY.NS','AUROPHARMA.NS','BALKRISIND.NS','BALRAMCHIN.NS','BEL.NS','BERGEPAINT.NS','BHEL.NS','BIOCON.NS','BOSCHLTD.NS','CGPOWER.NS',
    'CONCOR.NS','CROMPTON.NS','CUMMINSIND.NS','DABUR.NS','DEEPAKNTR.NS','ESCORTS.NS','GICRE.NS','GLENMARK.NS','GODREJCP.NS','GODREJPROP.NS',
    'GRANULES.NS','GUJGASLTD.NS','HAL.NS','HAPPSTMNDS.NS','HAVELLS.NS','IBREALEST.NS','IDEA.NS','INDHOTEL.NS','INDIGO.NS','INOXLEISUR.NS',
    'INTELLECT.NS','LALPATHLAB.NS','LAURUSLABS.NS','LUPIN.NS','MANAPPURAM.NS','MFSL.NS','MGL.NS','MRF.NS',
    'NAM-INDIA.NS','NAUKRI.NS','OBEROIRLTY.NS','OFSS.NS','PAGEIND.NS','PETRONET.NS','PFIZER.NS','POLICYBZR.NS','POWERINDIA.NS','PVRINOX.NS',
    'RAJESHEXPO.NS','RAMCOCEM.NS','RBLBANK.NS','SJVN.NS','TATACHEM.NS','TATAELXSI.NS','TATAPOWER.NS','TATACOMM.NS','TORNTPHARM.NS','TRENT.NS'
]

lookback_years = 5
retries = 3

end_date = datetime.now().date() - timedelta(days=1)
start_date = end_date - timedelta(days=lookback_years*365)

successful = 0
failed = []
total_samples = 0

print("=" * 70)
print("🇮🇳 INDIAN STOCK DATA DOWNLOADER - NSE")
print("=" * 70)
print(f"📅 Date Range: {start_date} to {end_date}")
print(f"📊 Total Tickers: {len(tickers)}")
print("=" * 70)

for i, ticker in enumerate(tickers, 1):
    name = ticker.replace('.NS', '')
    filename = f"{name}.csv"
    print(f"[{i:3d}/{len(tickers)}] {name:15s}...", end=" ", flush=True)
    
    n_attempts = 0
    while n_attempts < retries:
        try:
            # 🔥 FIX: Use Ticker object instead of download()
            stock = yf.Ticker(ticker)
            df = stock.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                auto_adjust=True  # Adjust for splits/dividends
            )
            
            if df.empty:
                print("❌ No data")
                failed.append(ticker)
                break

            # Reset index to get Date as column
            df = df.reset_index()
            
            # Clean column names (yfinance may use different casing)
            df.columns = [col.strip() for col in df.columns]
            
            # Rename to standard format
            if 'Date' not in df.columns and 'date' not in df.columns:
                df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
            elif 'date' in df.columns:
                df.rename(columns={'date': 'Date'}, inplace=True)
            
            # 🔥 CRITICAL: Force numeric dtypes
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
            df['High'] = pd.to_numeric(df['High'], errors='coerce')
            df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            
            # Select and reorder columns
            df = df[['Date','Open','High','Low','Close','Volume']]
            df['Name'] = name
            
            # Drop invalid rows
            df = df.dropna(subset=['Date', 'Close'])
            
            if len(df) < 100:
                print("❌ <100 rows")
                failed.append(ticker)
                break

            # Save with explicit float format
            df.to_csv(filename, index=False, float_format='%.4f')
            print(f"✅ {len(df):4d} days")
            successful += 1
            total_samples += len(df)
            break
            
        except Exception as e:
            error_msg = str(e)[:50].replace('\n', ' ')
            print(f"⚠️ {error_msg}...", end=" ", flush=True)
            n_attempts += 1
            time.sleep(2)  # Longer delay for API issues
            if n_attempts >= retries:
                print("❌ FAILED")
                failed.append(ticker)
    
    time.sleep(0.3)  # Rate limiting

print("\n" + "=" * 70)
print(f"✅ Successfully downloaded: {successful}/{len(tickers)} stocks")
print(f"📊 Total data points: {total_samples:,} rows")
if failed:
    print(f"❌ Failed tickers ({len(failed)}): {', '.join([t.replace('.NS','') for t in failed[:10]])}")
    if len(failed) > 10:
        print(f"   ... and {len(failed)-10} more")
print("=" * 70)

# 🔥 COMBINE ALL CSVs
print("\n📦 Combining all CSV files...")
csv_files = [f"{ticker.replace('.NS','')}.csv" for ticker in tickers 
             if Path(f"{ticker.replace('.NS','')}.csv").exists()]

all_data = []
for fname in csv_files:
    try:
        df = pd.read_csv(fname)
        # Verify numeric dtypes
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        all_data.append(df)
    except Exception as e:
        print(f"⚠️ Skipping {fname}: {e}")
        continue

if all_data:
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['Name', 'Date'])
    combined = combined.dropna(subset=['Close'])  # Final cleanup
    
    outfile = 'all_indian_stocks_5y.csv'
    combined.to_csv(outfile, index=False, float_format='%.4f')
    
    print(f"\n✅ Combined CSV created: {outfile}")
    print(f"   📊 Total rows: {len(combined):,}")
    print(f"   📈 Unique stocks: {combined['Name'].nunique()}")
    print(f"   📅 Date range: {combined['Date'].min()} to {combined['Date'].max()}")
    
    # Show sample data
    print(f"\n📋 Sample data (first stock):")
    sample = combined[combined['Name'] == combined['Name'].iloc[0]].head(3)
    print(sample.to_string(index=False))
    print("=" * 70)
else:
    print("❌ No data to combine")
