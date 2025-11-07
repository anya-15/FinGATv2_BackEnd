"""
🔮 SINGLE STOCK PREDICTOR
Get prediction for any stock from pre-generated CSV files
"""

import pandas as pd
from datetime import datetime
import os
import glob
import sys
from pathlib import Path
from difflib import get_close_matches

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class SingleStockPredictor:
    def __init__(self, predictions_folder='predictions'):
        """Initialize predictor with CSV predictions"""
        self.predictions_folder = predictions_folder
        
        print("🔮 Loading Stock Predictions...")
        print("="*60)
        
        # Find latest prediction file (try multiple patterns)
        csv_files = glob.glob(f"{predictions_folder}/predictions_*.csv")
        if not csv_files:
            # Try alternative pattern (up_only, top20, etc.)
            csv_files = glob.glob(f"{predictions_folder}/*_20*.csv")
        
        if not csv_files:
            print(f"❌ ERROR: No prediction files found in '{predictions_folder}/'")
            print("💡 Run 'python predict_now.py' first to generate predictions")
            raise FileNotFoundError(f"No prediction files in {predictions_folder}/")
        
        # Get most recent file
        self.latest_file = max(csv_files, key=os.path.getctime)
        
        # Extract date from filename
        filename = os.path.basename(self.latest_file)
        self.prediction_date = filename.replace('predictions_', '').replace('.csv', '')
        
        # Load predictions
        self.predictions_df = pd.read_csv(self.latest_file)
        
        # Add Rank column if not present
        if 'Rank' not in self.predictions_df.columns:
            self.predictions_df['Rank'] = range(1, len(self.predictions_df) + 1)
        
        print(f"✅ Loaded predictions from: {filename}")
        print(f"📅 Prediction date: {self.prediction_date}")
        print(f"📊 Total stocks: {len(self.predictions_df)}")
        print("="*60)
        
    def predict_stock(self, ticker):
        """Get prediction for a single stock"""
        
        # Validate ticker
        ticker = ticker.upper()
        stock_data = self.predictions_df[self.predictions_df['Ticker'] == ticker]
        
        if stock_data.empty:
            print(f"\n❌ ERROR: Ticker '{ticker}' not found in predictions!")
            
            # Fuzzy search for similar tickers
            all_tickers = self.predictions_df['Ticker'].tolist()
            similar = get_close_matches(ticker, all_tickers, n=5, cutoff=0.6)
            
            if similar:
                print(f"\n💡 Did you mean one of these?")
                print(f"   {', '.join(similar)}")
            else:
                # Fallback to substring search
                substring_matches = [t for t in all_tickers if ticker in t][:5]
                if substring_matches:
                    print(f"\n💡 Stocks containing '{ticker}':")
                    print(f"   {', '.join(substring_matches)}")
                else:
                    print(f"\n💡 Type 'list' to see all available stocks")
            
            return None
        
        # Extract stock data
        stock = stock_data.iloc[0]
        
        return {
            'ticker': str(stock['Ticker']),
            'sector': str(stock['Sector']),
            'direction': str(stock['Direction']),
            'confidence': float(stock['Confidence_%']),
            'up_probability': float(stock['UP_Probability']) * 100,
            'expected_return': float(stock['Expected_Return_%']),
            'ranking_score': float(stock['Ranking_Score']),
            'rank': int(stock['Rank']),
            'total_stocks': int(len(self.predictions_df))
        }
    
    def display_prediction(self, result):
        """Display prediction in a nice format"""
        if result is None:
            return
        
        print("\n" + "="*60)
        print("🎯 STOCK PREDICTION")
        print("="*60)
        print(f"📊 Stock: {result['ticker']}")
        print(f"🏢 Sector: {result['sector']}")
        print(f"📅 Prediction for: Next trading day after {self.prediction_date}")
        print("-"*60)
        
        # Direction with emoji
        if result['direction'] == "UP":
            direction_emoji = "⬆️ 📈"
            direction_color = "🟢"
        else:
            direction_emoji = "⬇️ 📉"
            direction_color = "🔴"
        
        print(f"{direction_color} Direction: {result['direction']} {direction_emoji}")
        print(f"💪 Confidence: {result['confidence']:.1f}%")
        print(f"📊 UP Probability: {result['up_probability']:.1f}%")
        print(f"💰 Expected Return: {result['expected_return']:.2f}% (weekly)")
        print(f"🏆 Rank: #{result['rank']} out of {result['total_stocks']} stocks")
        print(f"⭐ Ranking Score: {result['ranking_score']:.4f}")
        print("="*60)
        
        # Interpretation
        print("\n💡 INTERPRETATION:")
        
        # Confidence strength
        if result['confidence'] > 50:
            strength = "🔥 VERY HIGH"
        elif result['confidence'] > 35:
            strength = "✅ HIGH"
        elif result['confidence'] > 20:
            strength = "⚠️ MODERATE"
        else:
            strength = "❌ LOW"
        
        print(f"   Confidence Level: {strength}")
        
        # Quality based on rank
        if result['rank'] <= 5:
            quality = "🏆 EXCELLENT (Top 5!)"
        elif result['rank'] <= 10:
            quality = "🌟 EXCELLENT (Top 10!)"
        elif result['rank'] <= 20:
            quality = "✅ VERY GOOD (Top 20)"
        elif result['rank'] <= 50:
            quality = "👍 GOOD (Top 50)"
        else:
            quality = "📊 AVERAGE"
        
        print(f"   Quality Rank: {quality}")
        
        # Trading suggestion
        print("\n📈 TRADING SUGGESTION:")
        if result['direction'] == "UP" and result['confidence'] > 35 and result['rank'] <= 20:
            print("   ✅ STRONG BUY - High confidence + Top 20 rank")
            print("   💰 Suggested allocation: 10-15% of portfolio")
        elif result['direction'] == "UP" and result['confidence'] > 20:
            print("   ✅ BUY - Moderate confidence upward prediction")
            print("   💰 Suggested allocation: 5-10% of portfolio")
        elif result['direction'] == "UP":
            print("   ⚠️ WEAK BUY - Low confidence, higher risk")
            print("   💰 Suggested allocation: 2-5% of portfolio")
        elif result['direction'] == "DOWN" and result['confidence'] > 35:
            print("   ❌ STRONG SELL/AVOID - High confidence downward prediction")
            print("   🚫 Do NOT buy this stock")
        elif result['direction'] == "DOWN" and result['confidence'] > 20:
            print("   ❌ SELL/AVOID - Moderate confidence downward prediction")
            print("   🚫 Avoid or reduce position")
        else:
            print("   ⚠️ NEUTRAL - Low confidence, uncertain direction")
            print("   🤔 Wait for better signal")
        
        # Risk assessment
        print("\n⚠️ RISK ASSESSMENT:")
        if result['confidence'] > 40:
            risk = "LOW - High conviction"
        elif result['confidence'] > 25:
            risk = "MODERATE - Reasonable confidence"
        else:
            risk = "HIGH - Low confidence signal"
        print(f"   Risk Level: {risk}")
        
        # Investment calculation
        if result['direction'] == "UP" and result['rank'] <= 50:
            print("\n💰 INVESTMENT CALCULATOR:")
            for amount in [10000, 50000, 100000, 200000]:
                expected_profit = amount * (result['expected_return'] / 100)
                print(f"   ₹{amount:,} → Expected profit: ₹{expected_profit:,.0f} (weekly)")
        
        print("="*60 + "\n")
    
    def list_stocks(self):
        """List all available stocks"""
        print("\n📋 AVAILABLE STOCKS:")
        print("="*60)
        
        tickers = sorted(self.predictions_df['Ticker'].tolist())
        
        for i in range(0, len(tickers), 6):
            row = tickers[i:i+6]
            print("  " + ", ".join(f"{t:<12}" for t in row))
        
        print("="*60)
        print(f"Total: {len(tickers)} stocks")
        print("="*60 + "\n")
    
    def show_top_picks(self, n=10):
        """Show top N stock picks"""
        print(f"\n🏆 TOP {n} STOCK PICKS:")
        print("="*80)
        print(f"{'Rank':<6} {'Ticker':<12} {'Dir':<5} {'Conf':<8} {'Return':<10} {'Sector':<20}")
        print("-"*80)
        
        top_stocks = self.predictions_df.head(n)
        
        for idx, row in top_stocks.iterrows():
            direction_emoji = "⬆️" if row['Direction'] == "UP" else "⬇️"
            conf_bar = "█" * int(row['Confidence_%'] / 10)
            print(f"#{row['Rank']:<5d} {row['Ticker']:<12} {direction_emoji} {row['Direction']:<3} "
                  f"{row['Confidence_%']:5.1f}%  {row['Expected_Return_%']:+6.2f}%    {row['Sector']:<20}")
        
        print("="*80 + "\n")
    
    def filter_by_sector(self, sector):
        """Show stocks from a specific sector"""
        sector_stocks = self.predictions_df[
            self.predictions_df['Sector'].str.contains(sector, case=False, na=False)
        ].head(20)
        
        if sector_stocks.empty:
            print(f"\n❌ No stocks found in sector: {sector}")
            print("\n💡 Available sectors:")
            sectors = sorted(self.predictions_df['Sector'].unique())
            for s in sectors:
                count = len(self.predictions_df[self.predictions_df['Sector'] == s])
                print(f"   • {s} ({count} stocks)")
            return
        
        print(f"\n📊 STOCKS IN {sector.upper()} SECTOR:")
        print("="*80)
        print(f"{'Rank':<6} {'Ticker':<12} {'Dir':<5} {'Conf':<8} {'Return':<10}")
        print("-"*80)
        
        for idx, row in sector_stocks.iterrows():
            direction_emoji = "⬆️" if row['Direction'] == "UP" else "⬇️"
            print(f"#{row['Rank']:<5d} {row['Ticker']:<12} {direction_emoji} {row['Direction']:<3} "
                  f"{row['Confidence_%']:5.1f}%  {row['Expected_Return_%']:+6.2f}%")
        
        print("="*80 + "\n")
    
    def compare_stocks(self, tickers):
        """Compare multiple stocks side by side"""
        print(f"\n📊 STOCK COMPARISON:")
        print("="*100)
        print(f"{'Ticker':<12} {'Sector':<20} {'Dir':<5} {'Conf':<8} {'Return':<10} {'Rank':<8}")
        print("-"*100)
        
        for ticker in tickers:
            ticker = ticker.upper().strip()
            stock_data = self.predictions_df[self.predictions_df['Ticker'] == ticker]
            
            if stock_data.empty:
                print(f"{ticker:<12} ❌ Not found")
                continue
            
            row = stock_data.iloc[0]
            direction_emoji = "⬆️" if row['Direction'] == "UP" else "⬇️"
            print(f"{row['Ticker']:<12} {row['Sector']:<20} {direction_emoji} {row['Direction']:<3} "
                  f"{row['Confidence_%']:5.1f}%  {row['Expected_Return_%']:+6.2f}%    #{row['Rank']:<6d}")
        
        print("="*100 + "\n")
    
    def show_sectors(self):
        """Show all available sectors with stock counts"""
        print("\n📊 AVAILABLE SECTORS:")
        print("="*60)
        
        sector_counts = self.predictions_df['Sector'].value_counts()
        
        for sector, count in sector_counts.items():
            print(f"   • {sector:<30} ({count:2d} stocks)")
        
        print("="*60)
        print(f"Total sectors: {len(sector_counts)}")
        print("="*60 + "\n")


def get_single_stock_prediction(ticker: str, predictions_folder: str = 'predictions'):
    """
    API-friendly function to get prediction for a single stock
    
    Args:
        ticker: Stock ticker symbol
        predictions_folder: Path to predictions folder
    
    Returns:
        dict: Prediction data or None if not found
    """
    try:
        predictor = SingleStockPredictor(predictions_folder)
        return predictor.predict_stock(ticker)
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    """Main interactive loop"""
    
    print("\n" + "="*60)
    print("🎯 FINGAT SINGLE STOCK PREDICTION TOOL")
    print("="*60)
    print("📊 Get AI predictions for any stock")
    print("🤖 Powered by Graph Neural Networks (GATv2)")
    print("="*60 + "\n")
    
    try:
        predictor = SingleStockPredictor('predictions')
    except Exception as e:
        print(f"❌ Error loading predictor: {e}")
        return
    
    print("\n💡 COMMANDS:")
    print("   • Type stock ticker (e.g., INFY, TCS, RELIANCE)")
    print("   • Type 'list' to see all available stocks")
    print("   • Type 'top' or 'top5/top10/top20' to see top picks")
    print("   • Type 'sectors' to see all sectors")
    print("   • Type 'sector:<name>' to filter by sector (e.g., sector:tech)")
    print("   • Type 'compare:<ticker1>,<ticker2>,...' to compare stocks")
    print("   • Type 'help' to see this menu again")
    print("   • Type 'exit' or 'quit' to close")
    print("-"*60 + "\n")
    
    while True:
        try:
            # Get user input
            command = input("🔍 Enter command or ticker: ").strip()
            
            # Check for exit
            if command.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye! Happy trading! 🚀\n")
                break
            
            # Skip empty input
            if not command:
                continue
            
            # Check for help command
            if command.lower() == 'help':
                print("\n💡 AVAILABLE COMMANDS:")
                print("   • <ticker>           - Get prediction for a stock")
                print("   • list               - List all available stocks")
                print("   • top/top5/top10/top20 - Show top stock picks")
                print("   • sectors            - Show all sectors")
                print("   • sector:<name>      - Filter by sector (e.g., sector:tech)")
                print("   • compare:<t1>,<t2>  - Compare stocks (e.g., compare:INFY,TCS)")
                print("   • help               - Show this menu")
                print("   • exit/quit          - Exit program\n")
                continue
            
            # Check for list command
            if command.lower() == 'list':
                predictor.list_stocks()
                continue
            
            # Check for sectors command
            if command.lower() == 'sectors':
                predictor.show_sectors()
                continue
            
            # Check for sector filter
            if command.lower().startswith('sector:'):
                sector = command.split(':', 1)[1].strip()
                predictor.filter_by_sector(sector)
                continue
            
            # Check for compare command
            if command.lower().startswith('compare:'):
                tickers_str = command.split(':', 1)[1]
                tickers = [t.strip() for t in tickers_str.split(',')]
                predictor.compare_stocks(tickers)
                continue
            
            # Check for top command
            if command.lower() in ['top', 'top5', 'top10', 'top20']:
                n = 10  # default
                if 'top5' in command.lower():
                    n = 5
                elif 'top20' in command.lower():
                    n = 20
                predictor.show_top_picks(n)
                continue
            
            # Assume it's a ticker
            result = predictor.predict_stock(command)
            
            # Display result
            predictor.display_prediction(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Happy trading! 🚀\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
