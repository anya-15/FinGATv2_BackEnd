"""
Populate Database with Stock Metadata
Reads stock info from CSV files and populates PostgreSQL
EXPANDED VERSION - Covers all major NSE/BSE stocks
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from sqlalchemy.orm import Session

from app.db.database import SessionLocal, init_db
from app.db.models import Stock
from app.config import settings


# COMPREHENSIVE NSE/BSE sector mapping (148+ stocks)
SECTOR_MAPPING = {
    # ============================================
    # TECHNOLOGY & IT SERVICES
    # ============================================
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
    'HAPPSTMNDS': ('Technology', 'IT Services'),
    'INTELLECT': ('Technology', 'Software'),
    'OFSS': ('Technology', 'Financial Software'),
    'NAUKRI': ('Technology', 'Internet Services'),
    
    # ============================================
    # BANKING & FINANCE
    # ============================================
    'HDFCBANK': ('Finance', 'Private Banking'),
    'ICICIBANK': ('Finance', 'Private Banking'),
    'SBIN': ('Finance', 'Public Banking'),
    'AXISBANK': ('Finance', 'Private Banking'),
    'KOTAKBANK': ('Finance', 'Private Banking'),
    'INDUSINDBK': ('Finance', 'Private Banking'),
    'BANKBARODA': ('Finance', 'Public Banking'),
    'PNB': ('Finance', 'Public Banking'),
    'CANBK': ('Finance', 'Public Banking'),
    'UNIONBANK': ('Finance', 'Public Banking'),
    'UCOBANK': ('Finance', 'Public Banking'),
    'IDBI': ('Finance', 'Public Banking'),
    'FEDERALBNK': ('Finance', 'Private Banking'),
    'IDFCFIRSTB': ('Finance', 'Private Banking'),
    'BANDHANBNK': ('Finance', 'Private Banking'),
    'RBLBANK': ('Finance', 'Private Banking'),
    'AUBANK': ('Finance', 'Private Banking'),
    'YESBANK': ('Finance', 'Private Banking'),
    
    # ============================================
    # NBFC & FINANCIAL SERVICES
    # ============================================
    'BAJFINANCE': ('Finance', 'NBFC'),
    'BAJAJFINSV': ('Finance', 'Financial Services'),
    'CHOLAFIN': ('Finance', 'NBFC'),
    'LICHSGFIN': ('Finance', 'Housing Finance'),
    'MFSL': ('Finance', 'Financial Services'),
    'MOTILALOFS': ('Finance', 'Stock Broking'),
    'PFC': ('Finance', 'Infrastructure Finance'),
    'RECLTD': ('Finance', 'Infrastructure Finance'),
    'MANAPPURAM': ('Finance', 'Gold Loan NBFC'),
    'MUTHOOTFIN': ('Finance', 'Gold Loan NBFC'),
    'SUNDARMFIN': ('Finance', 'NBFC'),
    'SBICARD': ('Finance', 'Credit Cards'),
    'HDFCAMC': ('Finance', 'Asset Management'),
    
    # ============================================
    # INSURANCE
    # ============================================
    'HDFCLIFE': ('Finance', 'Life Insurance'),
    'SBILIFE': ('Finance', 'Life Insurance'),
    'ICICIPRULI': ('Finance', 'Life Insurance'),
    'ICICIGI': ('Finance', 'General Insurance'),
    'GICRE': ('Finance', 'Reinsurance'),
    'POLICYBZR': ('Finance', 'Insurance Distribution'),
    
    # ============================================
    # ENERGY - OIL & GAS
    # ============================================
    'RELIANCE': ('Energy', 'Integrated Oil & Gas'),
    'ONGC': ('Energy', 'Exploration & Production'),
    'IOC': ('Energy', 'Oil Refining & Marketing'),
    'BPCL': ('Energy', 'Oil Refining & Marketing'),
    'GAIL': ('Energy', 'Natural Gas'),
    'PETRONET': ('Energy', 'LNG'),
    'OIL': ('Energy', 'Exploration & Production'),
    'GUJGASLTD': ('Energy', 'Gas Distribution'),
    'MGL': ('Energy', 'Gas Distribution'),
    
    # ============================================
    # POWER & UTILITIES
    # ============================================
    'NTPC': ('Utilities', 'Power Generation'),
    'POWERGRID': ('Utilities', 'Power Transmission'),
    'ADANIGREEN': ('Utilities', 'Renewable Energy'),
    'ADANIPOWER': ('Utilities', 'Power Generation'),
    'SJVN': ('Utilities', 'Hydro Power'),
    'POWERINDIA': ('Utilities', 'Power Equipment'),
    
    # ============================================
    # AUTOMOTIVE
    # ============================================
    'MARUTI': ('Automotive', 'Passenger Vehicles'),
    'TATAMOTORS': ('Automotive', 'Commercial & Passenger Vehicles'),
    'M&M': ('Automotive', 'Tractors & Vehicles'),
    'BAJAJ-AUTO': ('Automotive', 'Two Wheelers'),
    'HEROMOTOCO': ('Automotive', 'Two Wheelers'),
    'EICHERMOT': ('Automotive', 'Two Wheelers & Commercial'),
    'ESCORTS': ('Automotive', 'Tractors & Construction'),
    'ASHOKLEY': ('Automotive', 'Commercial Vehicles'),
    'BALKRISIND': ('Automotive', 'Tires'),
    'MRF': ('Automotive', 'Tires'),
    'BOSCHLTD': ('Automotive', 'Auto Components'),
    'MOTHERSON': ('Automotive', 'Auto Components'),
    
    # ============================================
    # PHARMACEUTICALS
    # ============================================
    'SUNPHARMA': ('Pharmaceuticals', 'Drug Manufacturing'),
    'DRREDDY': ('Pharmaceuticals', 'Drug Manufacturing'),
    'CIPLA': ('Pharmaceuticals', 'Drug Manufacturing'),
    'DIVISLAB': ('Pharmaceuticals', 'Drug Manufacturing'),
    'LUPIN': ('Pharmaceuticals', 'Drug Manufacturing'),
    'AUROPHARMA': ('Pharmaceuticals', 'Drug Manufacturing'),
    'BIOCON': ('Pharmaceuticals', 'Biopharmaceuticals'),
    'TORNTPHARM': ('Pharmaceuticals', 'Drug Manufacturing'),
    'ALKEM': ('Pharmaceuticals', 'Drug Manufacturing'),
    'GLENMARK': ('Pharmaceuticals', 'Drug Manufacturing'),
    'LAURUSLABS': ('Pharmaceuticals', 'API Manufacturing'),
    'GRANULES': ('Pharmaceuticals', 'API Manufacturing'),
    'PFIZER': ('Pharmaceuticals', 'Drug Manufacturing'),
    'ABBOTINDIA': ('Pharmaceuticals', 'Drug Manufacturing'),
    'LALPATHLAB': ('Healthcare', 'Diagnostics'),
    'APOLLOHOSP': ('Healthcare', 'Hospitals'),
    
    # ============================================
    # CONSUMER GOODS - FMCG
    # ============================================
    'HINDUNILVR': ('Consumer Goods', 'FMCG'),
    'ITC': ('Consumer Goods', 'FMCG & Tobacco'),
    'NESTLEIND': ('Consumer Goods', 'Foods'),
    'BRITANNIA': ('Consumer Goods', 'Foods'),
    'DABUR': ('Consumer Goods', 'Personal Care'),
    'GODREJCP': ('Consumer Goods', 'Personal Care'),
    'MARICO': ('Consumer Goods', 'Personal Care'),
    'COLPAL': ('Consumer Goods', 'Personal Care'),
    'TATACONSUM': ('Consumer Goods', 'Foods & Beverages'),
    'EMAMILTD': ('Consumer Goods', 'Personal Care'),
    'JUBLPHARMA': ('Consumer Goods', 'Confectionery'),
    'JUBLFOOD': ('Consumer Goods', 'QSR'),
    
    # ============================================
    # METALS & MINING
    # ============================================
    'TATASTEEL': ('Metals', 'Steel'),
    'JSWSTEEL': ('Metals', 'Steel'),
    'HINDALCO': ('Metals', 'Aluminum & Copper'),
    'VEDL': ('Metals', 'Diversified Mining'),
    'COALINDIA': ('Metals', 'Coal Mining'),
    'JINDALSTEL': ('Metals', 'Steel'),
    'SAIL': ('Metals', 'Steel'),
    'NMDC': ('Metals', 'Iron Ore Mining'),
    'NATIONALUM': ('Metals', 'Aluminum'),
    'HINDZINC': ('Metals', 'Zinc & Lead'),
    
    # ============================================
    # CEMENT & CONSTRUCTION
    # ============================================
    'ULTRACEMCO': ('Construction', 'Cement'),
    'AMBUJACEM': ('Construction', 'Cement'),
    'ACC': ('Construction', 'Cement'),
    'SHREECEM': ('Construction', 'Cement'),
    'RAMCOCEM': ('Construction', 'Cement'),
    'GRASIM': ('Construction', 'Cement & Textiles'),
    'LT': ('Construction', 'Engineering & Construction'),
    
    # ============================================
    # REAL ESTATE
    # ============================================
    'DLF': ('Real Estate', 'Residential & Commercial'),
    'GODREJPROP': ('Real Estate', 'Residential & Commercial'),
    'OBEROIRLTY': ('Real Estate', 'Residential & Commercial'),
    'IBREALEST': ('Real Estate', 'Commercial'),
    'PRESTIGE': ('Real Estate', 'Residential & Commercial'),
    
    # ============================================
    # RETAIL & CONSUMER
    # ============================================
    'DMART': ('Retail', 'Supermarkets'),
    'TRENT': ('Retail', 'Fashion Retail'),
    'TITAN': ('Consumer Goods', 'Jewelry & Watches'),
    'PAGEIND': ('Consumer Goods', 'Footwear'),
    
    # ============================================
    # TELECOM
    # ============================================
    'BHARTIARTL': ('Telecom', 'Mobile Telecom'),
    'IDEA': ('Telecom', 'Mobile Telecom'),
    'TATACOMM': ('Telecom', 'Enterprise Telecom'),
    
    # ============================================
    # MEDIA & ENTERTAINMENT
    # ============================================
    'PVRINOX': ('Entertainment', 'Cinema Exhibition'),
    'INOXLEISUR': ('Entertainment', 'Cinema Exhibition'),
    'ZEEL': ('Media', 'Broadcasting'),
    'SUNTV': ('Media', 'Broadcasting'),
    
    # ============================================
    # PAINTS & CHEMICALS
    # ============================================
    'ASIANPAINT': ('Chemicals', 'Paints'),
    'BERGEPAINT': ('Chemicals', 'Paints'),
    'KANSAINER': ('Chemicals', 'Paints'),
    'PIDILITIND': ('Chemicals', 'Adhesives'),
    'AARTIIND': ('Chemicals', 'Specialty Chemicals'),
    'DEEPAKNTR': ('Chemicals', 'Specialty Chemicals'),
    'SRF': ('Chemicals', 'Specialty Chemicals'),
    'TATACHEM': ('Chemicals', 'Basic Chemicals'),
    'UPL': ('Chemicals', 'Agrochemicals'),
    'BALRAMCHIN': ('Chemicals', 'Basic Chemicals'),
    
    # ============================================
    # ELECTRONICS & ELECTRICALS
    # ============================================
    'HAVELLS': ('Consumer Goods', 'Electrical Equipment'),
    'CROMPTON': ('Consumer Goods', 'Electrical Equipment'),
    'VGUARD': ('Consumer Goods', 'Electrical Equipment'),
    'POLYCAB': ('Industrials', 'Cables'),
    'KEI': ('Industrials', 'Cables'),
    
    # ============================================
    # DIVERSIFIED
    # ============================================
    'ADANIENT': ('Diversified', 'Conglomerate'),
    'ADANIPORTS': ('Logistics', 'Port Operations'),
    'BAJAJHLDNG': ('Finance', 'Holding Company'),
    
    # ============================================
    # ENGINEERING & CAPITAL GOODS
    # ============================================
    'BEL': ('Defense', 'Electronics'),
    'HAL': ('Defense', 'Aerospace'),
    'BHEL': ('Industrials', 'Power Equipment'),
    'CUMMINSIND': ('Industrials', 'Engines'),
    'ABB': ('Industrials', 'Electrical Equipment'),
    'SIEMENS': ('Industrials', 'Electrical Equipment'),
    'THERMAX': ('Industrials', 'Boilers & Heaters'),
    'CGPOWER': ('Industrials', 'Electrical Equipment'),
    
    # ============================================
    # LOGISTICS & TRANSPORT
    # ============================================
    'CONCOR': ('Logistics', 'Container Logistics'),
    'VRL': ('Logistics', 'Road Transport'),
    'INDIGO': ('Airlines', 'Aviation'),
    'INDHOTEL': ('Hospitality', 'Hotels'),
    
    # ============================================
    # TEXTILES
    # ============================================
    'RAYMOND': ('Textiles', 'Fabrics'),
    'VARDHACRLC': ('Textiles', 'Yarn'),
    'WELSPUNIND': ('Textiles', 'Home Textiles'),
    'TRIDENT': ('Textiles', 'Home Textiles'),
    
    # ============================================
    # SPECIALTY / OTHER
    # ============================================
    'ADVENZYMES': ('Industrials', 'Enzymes'),
    'RAJESHEXPO': ('Gems & Jewelry', 'Diamond Processing'),
    'PEL': ('Consumer Goods', 'Home Appliances'),
    'WHIRLPOOL': ('Consumer Goods', 'Home Appliances'),
    'VOLTAS': ('Consumer Goods', 'Air Conditioning'),
    'NAM-INDIA': ('Industrials', 'Bearings'),
}


def populate_stocks():
    """
    Populate database with stock information from CSV files
    """
    
    print("=" * 60)
    print("📊 Populating Database with Stock Metadata")
    print("=" * 60)
    
    # Initialize database tables
    init_db()
    
    data_dir = Path(settings.DATA_PATH)
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    # Get all CSV files
    stock_files = list(data_dir.glob("*.csv"))
    
    if len(stock_files) == 0:
        print(f"❌ No stock files found in {data_dir}/")
        return False
    
    print(f"\n📈 Found {len(stock_files)} stock CSV files")
    
    db = SessionLocal()
    added_count = 0
    updated_count = 0
    error_count = 0
    total_files = len(stock_files)
    
    # Track sector statistics
    sector_stats = {}
    
    try:
        for idx, csv_file in enumerate(stock_files, 1):
            ticker = csv_file.stem
            
            try:
                # Read CSV to get company name
                df = pd.read_csv(csv_file)
                
                # Get company name from CSV (if exists)
                if 'name' in df.columns:
                    company_name = df['name'].iloc[0]
                else:
                    company_name = ticker
                
                # Get latest price
                if 'Close' in df.columns:
                    current_price = float(df['Close'].iloc[-1])
                else:
                    current_price = None
                
                # Get sector from mapping
                sector, industry = SECTOR_MAPPING.get(ticker, ('Unknown', 'Unknown'))
                
                # Track sector stats
                if sector != 'Unknown':
                    sector_stats[sector] = sector_stats.get(sector, 0) + 1
                
                # Check if stock exists
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
                
                # Truncate company name for display
                display_name = company_name[:35] if len(company_name) > 35 else company_name
                print(f"[{idx}/{total_files}] {status} {ticker:15s} - {display_name:35s} ({sector})")
            
            except Exception as e:
                print(f"[{idx}/{total_files}] ❌ {ticker:15s} - Error: {str(e)[:30]}")
                error_count += 1
                continue
        
        # Commit all changes
        db.commit()
        
        print("\n" + "=" * 60)
        print("✅ Database Population Complete")
        print(f"   Added: {added_count}")
        print(f"   Updated: {updated_count}")
        print(f"   Errors: {error_count}")
        print(f"   Total: {added_count + updated_count}")
        print("\n📊 Sector Distribution:")
        for sector, count in sorted(sector_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"   {sector:25s}: {count:3d} stocks")
        print("=" * 60)
        
        return True
    
    except Exception as e:
        print(f"\n❌ Database error: {e}")
        db.rollback()
        return False
    
    finally:
        db.close()


if __name__ == '__main__':
    success = populate_stocks()
    sys.exit(0 if success else 1)
