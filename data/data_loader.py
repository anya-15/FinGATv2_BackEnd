import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FinancialDataCollector:
    """Load Indian stock data from local CSV files"""
    
    def __init__(self, csv_folder_path: str, max_stocks: int = 500):
        self.csv_folder_path = csv_folder_path
        self.max_stocks = max_stocks
        
    def collect_stock_data(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """Load stock data from CSV files"""
        
        if not os.path.exists(self.csv_folder_path):
            raise FileNotFoundError(f"Folder not found: {self.csv_folder_path}")
        
        all_data = {}
        csv_files = [f for f in os.listdir(self.csv_folder_path) if f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.csv_folder_path}")
        
        csv_files = csv_files[:self.max_stocks]
        print(f"Loading {len(csv_files)} CSV files...")
        
        for filename in csv_files:
            try:
                filepath = os.path.join(self.csv_folder_path, filename)
                
                base_name = os.path.splitext(filename)[0]
                if '_data' in base_name.lower():
                    ticker = base_name.replace('_data', '').replace('_DATA', '').upper()
                else:
                    ticker = base_name.upper()
                
                df = pd.read_csv(filepath)
                
                date_col = self._find_column(df, ['date', 'datetime', 'timestamp'])
                close_col = self._find_column(df, ['close', 'adj_close', 'adjusted_close'])
                open_col = self._find_column(df, ['open'])
                high_col = self._find_column(df, ['high'])
                low_col = self._find_column(df, ['low'])
                volume_col = self._find_column(df, ['volume', 'vol'])
                
                if not date_col or not close_col:
                    print(f"Warning: {ticker} missing required columns")
                    continue
                
                column_mapping = {date_col: 'Date', close_col: 'Close'}
                if open_col:
                    column_mapping[open_col] = 'Open'
                if high_col:
                    column_mapping[high_col] = 'High'
                if low_col:
                    column_mapping[low_col] = 'Low'
                if volume_col:
                    column_mapping[volume_col] = 'Volume'
                
                df = df.rename(columns=column_mapping)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').dropna(subset=['Close'])
                
                # ✅ CRITICAL: Need more history now (60+ days instead of 50)
                if len(df) < 60:
                    print(f"Warning: {ticker} insufficient data ({len(df)} rows)")
                    continue
                
                all_data[ticker] = df
                print(f"✓ Loaded {ticker}: {len(df)} rows")
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid stock data could be loaded")
        
        sectors = self._create_sector_mapping(list(all_data.keys()))
        return all_data, sectors
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        for col in df.columns:
            if col.lower() in [name.lower() for name in possible_names]:
                return col
        return None
    
    def _create_sector_mapping(self, tickers: List[str]) -> Dict[str, str]:
        """Indian stock sector mappings"""
        
        sector_map = {
            # Banks & Financial Services
            'HDFCBANK': 'Banks', 'ICICIBANK': 'Banks', 'SBIN': 'Banks',
            'KOTAKBANK': 'Banks', 'AXISBANK': 'Banks', 'INDUSINDBK': 'Banks',
            'BANKBARODA': 'Banks', 'PNB': 'Banks', 'UNIONBANK': 'Banks',
            'CANBK': 'Banks', 'IDFCFIRSTB': 'Banks', 'FEDERALBNK': 'Banks',
            'RBLBANK': 'Banks', 'BANDHANBNK': 'Banks', 'AUBANK': 'Banks',
            
            # Finance & NBFCs
            'BAJFINANCE': 'Finance', 'HDFCLIFE': 'Finance', 'SBILIFE': 'Finance',
            'BAJAJFINSV': 'Finance', 'ICICIGI': 'Finance', 'HDFCAMC': 'Finance',
            'MUTHOOTFIN': 'Finance', 'CHOLAFIN': 'Finance', 'LICHSGFIN': 'Finance',
            'PNBHOUSING': 'Finance', 'RECLTD': 'Finance', 'PFC': 'Finance',
            'SHRIRAMFIN': 'Finance', 'HUDCO': 'Finance', 'IRFC': 'Finance',
            
            # IT & Software
            'TCS': 'Software & IT Services', 'INFY': 'Software & IT Services',
            'WIPRO': 'Software & IT Services', 'HCLTECH': 'Software & IT Services',
            'TECHM': 'Software & IT Services', 'LTI': 'Software & IT Services',
            'MINDTREE': 'Software & IT Services', 'MPHASIS': 'Software & IT Services',
            'COFORGE': 'Software & IT Services', 'LTTS': 'Software & IT Services',
            'PERSISTENT': 'Software & IT Services', 'OFSS': 'Software & IT Services',
            'TATAELXSI': 'Software & IT Services', 'INTELLECT': 'Software & IT Services',
            
            # FMCG
            'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'NESTLEIND': 'FMCG',
            'BRITANNIA': 'FMCG', 'DABUR': 'FMCG', 'MARICO': 'FMCG',
            'GODREJCP': 'FMCG', 'COLPAL': 'FMCG', 'TATACONSUM': 'FMCG',
            
            # Automobile
            'MARUTI': 'Automobile & Ancillaries', 'TATAMOTORS': 'Automobile & Ancillaries',
            'M&M': 'Automobile & Ancillaries', 'BAJAJ-AUTO': 'Automobile & Ancillaries',
            'HEROMOTOCO': 'Automobile & Ancillaries', 'EICHERMOT': 'Automobile & Ancillaries',
            'TVSMOTOR': 'Automobile & Ancillaries', 'ASHOKLEY': 'Automobile & Ancillaries',
            'BALKRISIND': 'Automobile & Ancillaries', 'APOLLOTYRE': 'Automobile & Ancillaries',
            'MRF': 'Automobile & Ancillaries', 'MOTHERSON': 'Automobile & Ancillaries',
            'BOSCHLTD': 'Automobile & Ancillaries', 'ESCORTS': 'Automobile & Ancillaries',
            
            # Pharmaceuticals
            'SUNPHARMA': 'Healthcare', 'DRREDDY': 'Healthcare', 'CIPLA': 'Healthcare',
            'DIVISLAB': 'Healthcare', 'BIOCON': 'Healthcare', 'AUROPHARMA': 'Healthcare',
            'LUPIN': 'Healthcare', 'TORNTPHARM': 'Healthcare', 'ALKEM': 'Healthcare',
            'APOLLOHOSP': 'Healthcare', 'MAXHEALTH': 'Healthcare', 'FORTIS': 'Healthcare',
            
            # Oil & Gas
            'RELIANCE': 'Oil & Gas', 'ONGC': 'Oil & Gas', 'IOC': 'Oil & Gas',
            'BPCL': 'Oil & Gas', 'HINDPETRO': 'Oil & Gas', 'GAIL': 'Oil & Gas',
            'PETRONET': 'Oil & Gas', 'OIL': 'Oil & Gas', 'MGL': 'Oil & Gas', 'IGL': 'Oil & Gas',
            
            # Metals & Mining
            'TATASTEEL': 'Metals & Mining', 'JSWSTEEL': 'Metals & Mining',
            'HINDALCO': 'Metals & Mining', 'VEDL': 'Metals & Mining',
            'COALINDIA': 'Metals & Mining', 'JINDALSTEL': 'Metals & Mining',
            'SAIL': 'Metals & Mining', 'NMDC': 'Metals & Mining',
            
            # Cement
            'ULTRACEMCO': 'Cement & Construction', 'GRASIM': 'Cement & Construction',
            'AMBUJACEM': 'Cement & Construction', 'ACC': 'Cement & Construction',
            'SHREECEM': 'Cement & Construction', 'RAMCOCEM': 'Cement & Construction',
            
            # Power
            'POWERGRID': 'Power', 'NTPC': 'Power', 'ADANIPOWER': 'Power',
            'TATAPOWER': 'Power', 'ADANIGREEN': 'Power', 'TORNTPOWER': 'Power',
            
            # Telecom
            'BHARTIARTL': 'Communication Services', 'IDEA': 'Communication Services',
            'TATACOMM': 'Communication Services',
            
            # Chemicals
            'UPL': 'Chemicals', 'SRF': 'Chemicals', 'ATUL': 'Chemicals',
            'PIDILITIND': 'Chemicals', 'DEEPAKNTR': 'Chemicals', 'NAVINFLUOR': 'Chemicals',
            'AARTI': 'Chemicals', 'AARTIIND': 'Chemicals', 'GRANULES': 'Chemicals',
            
            # Industrials
            'LT': 'Industrials', 'SIEMENS': 'Industrials', 'ABB': 'Industrials',
            'BHARATFORG': 'Industrials', 'CUMMINSIND': 'Industrials',
            'THERMAX': 'Industrials', 'VOLTAS': 'Industrials', 'HAVELLS': 'Industrials',
            
            # Consumer Discretionary
            'TITAN': 'Consumer Discretionary', 'DMART': 'Consumer Discretionary',
            'TRENT': 'Consumer Discretionary', 'ABFRL': 'Consumer Discretionary',
            'JUBLFOOD': 'Consumer Discretionary', 'RELAXO': 'Consumer Discretionary',
            
            # Real Estate
            'DLF': 'Real Estate', 'GODREJPROP': 'Real Estate', 'OBEROIRLTY': 'Real Estate',
            'PHOENIXLTD': 'Real Estate', 'PRESTIGE': 'Real Estate',
            
            # Infrastructure
            'ADANIPORTS': 'Infrastructure & Logistics', 'CONCOR': 'Infrastructure & Logistics',
            'IRCTC': 'Infrastructure & Logistics', 'INDIGO': 'Infrastructure & Logistics',
            
            # Cards/Payments
            'SBICARD': 'Finance', 'PAYTM': 'Software & IT Services',
            
            # Paint
            'ASIANPAINT': 'Industrials', 'BERGER': 'Industrials',
        }
        
        sectors = {}
        sector_counts = {}
        
        for ticker in tickers:
            sector = sector_map.get(ticker, 'Other')
            sectors[ticker] = sector
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        print("\n🇮🇳 Indian Stock Market Sector Distribution:")
        print("=" * 60)
        for sector, count in sorted(sector_counts.items()):
            print(f"  {sector}: {count} stocks")
        print("=" * 60)
        
        return sectors


class FinancialFeatureEngineer:
    """LEAK-FREE feature engineering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.sector_encoder = LabelEncoder()
        
    def create_technical_features(self, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create technical indicators - REDUCED to match manifest checkpoint (7 features)"""
        
        features = {}
        
        for ticker, df in all_data.items():
            try:
                close_prices = df['Close']
                
                # ✅ MINIMAL FEATURE SET (7 features to match manifest checkpoint)
                # After mean+std aggregation: 7 × 2 = 14 features (no sector encoding)
                
                # 1. Returns
                features[f'{ticker}_returns'] = close_prices.pct_change()
                
                # 2. Volatility (20-day)
                features[f'{ticker}_volatility_20'] = features[f'{ticker}_returns'].rolling(20).std()
                
                # 3-4. Moving averages
                features[f'{ticker}_sma_5'] = close_prices.rolling(5).mean()
                features[f'{ticker}_sma_20'] = close_prices.rolling(20).mean()
                
                # 5. RSI
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-8)
                features[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))
                
                # 6. Momentum (5-day)
                features[f'{ticker}_momentum_5'] = close_prices / close_prices.shift(5) - 1
                
                # 7. Volume ratio
                if 'Volume' in df.columns:
                    volume = df['Volume']
                    features[f'{ticker}_volume_ratio'] = volume / volume.rolling(20).mean()
                else:
                    features[f'{ticker}_volume_ratio'] = pd.Series(1.0, index=close_prices.index)
                
                print(f"✓ Created features for {ticker}")
                
            except Exception as e:
                print(f"Error creating features for {ticker}: {e}")
                continue
        
        return pd.DataFrame(features)
    
    def create_graph_features(self, features: pd.DataFrame, sectors: Dict[str, str]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        ✅ LEAK-FREE: 40-day window with 5-day buffer
        Features: Days -60 to -11
        Buffer: Days -10 to -6 (not used)
        Target: Days -5 to -1
        """
        
        tickers = list(set([col.split('_')[0] for col in features.columns]))
        tickers = [t for t in tickers if t in sectors.keys()]
        
        node_features = []
        ticker_to_idx = {ticker: idx for idx, ticker in enumerate(tickers)}
        
        print("\n🚀 LEAK-FREE Feature Extraction:")
        print("=" * 60)
        
        for ticker in tickers:
            ticker_cols = [col for col in features.columns if col.startswith(f'{ticker}_')]
            ticker_data = features[ticker_cols].ffill().fillna(0)
            
            if len(ticker_data) > 60:
                # ✅ FIXED: Days -60 to -11 (50-day window, ends at -11)
                ticker_mean = ticker_data.iloc[-60:-11].mean(axis=0).values
                ticker_std = ticker_data.iloc[-60:-11].std(axis=0).values
                ticker_features = np.concatenate([ticker_mean, ticker_std])
                
                if len(node_features) == 0:
                    print(f"Example: {ticker}")
                    print(f"  ✅ Feature window: Days -60 to -11 (50 days)")
                    print(f"  ✅ Buffer: Days -10 to -6 (5 days, not used)")
                    print(f"  ✅ Target: Days -5 to -1 (5 days)")
                    print(f"  ✅ NO OVERLAP - NO LEAKAGE!")
                    print(f"  Features: {len(ticker_mean)} mean + {len(ticker_std)} std = {len(ticker_features)} total")
                    print(f"  ✅ NO SECTOR ENCODING (checkpoint doesn't use it)")
            else:
                ticker_features = np.zeros(len(ticker_cols) * 2)
            
            # ✅ NO SECTOR ENCODING - checkpoint was trained without it
            # Sector info is only used for graph construction (stock_to_sector)
            node_features.append(ticker_features)
        
        print("=" * 60)
        
        node_features = np.array(node_features)
        if node_features.std() > 0:
            node_features = self.scaler.fit_transform(node_features)
        
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # Create K-NN graph
        num_nodes = len(tickers)
        edges = []
        
        if num_nodes > 1:
            k_neighbors = min(15, num_nodes - 1)
            nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree')
            nbrs.fit(node_features.numpy())
            distances, indices = nbrs.kneighbors(node_features.numpy())
            
            for i in range(num_nodes):
                for j in indices[i]:
                    if i != j:
                        edges.append([i, j])
        
        if edges:
            edge_index = torch.tensor(list(set(map(tuple, edges))), dtype=torch.long).t().contiguous()
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        print(f"✅ Graph: {edge_index.size(1)} edges, {node_features.size(1)} features per node")
        
        return node_features, edge_index, ticker_to_idx


class FinancialDataset:
    """LEAK-FREE dataset"""
    
    def __init__(self, csv_folder_path: str, max_stocks: int = 550):
        self.csv_folder_path = csv_folder_path
        self.max_stocks = max_stocks
        self.collector = FinancialDataCollector(csv_folder_path, max_stocks)
        self.feature_engineer = FinancialFeatureEngineer()
        
    def prepare_dataset(self) -> Tuple[Data, Dict]:
        """Prepare dataset"""
        
        print("=" * 60)
        print("🚀 LEAK-FREE FinGAT 2025 - INDIAN MARKET")
        print("Expected: Honest 52-60% Accuracy")
        print("=" * 60)
        
        all_data, sectors = self.collector.collect_stock_data()
        
        print("\nCreating technical features...")
        features = self.feature_engineer.create_technical_features(all_data)
        
        print("\nBuilding LEAK-FREE graph...")
        node_features, edge_index, ticker_to_idx = self.feature_engineer.create_graph_features(features, sectors)
        
        print("\nCreating BALANCED targets...")
        targets = self._create_targets(features, list(ticker_to_idx.keys()))
        
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=targets,
            num_nodes=node_features.size(0)
        )
        
        metadata = {
            'ticker_to_idx': ticker_to_idx,
            'idx_to_ticker': {v: k for k, v in ticker_to_idx.items()},
            'sectors': sectors,
            'feature_names': features.columns.tolist(),
            'num_features': node_features.size(1),
            'num_stocks': len(all_data),
            'num_edges': edge_index.size(1),
            'sector_distribution': self._get_sector_distribution(sectors)
        }
        
        print(f"\n✓ Dataset ready:")
        print(f"  - {len(all_data)} stocks")
        print(f"  - {node_features.size(1)} features per node")
        print(f"  - {metadata['num_edges']} edges")
        
        # --- SECTOR-AWARE GRAPH ATTRIBUTES FOR FinGAT (API & batch prediction) ---
        # Create sector mapping
        sector_list = sorted(list(set(sectors.values())))
        sector_to_idx = {sector: i for i, sector in enumerate(sector_list)}
        ticker_list = list(ticker_to_idx.keys())
        
        # ✅ FIXED: stock_to_sector should be a 1D tensor mapping each stock to its sector ID
        # This is used as a "batch" tensor for pooling operations
        stock_to_sector_mapping = []
        for ticker in ticker_list:
            sector_name = sectors[ticker]
            sector_idx = sector_to_idx[sector_name]
            stock_to_sector_mapping.append(sector_idx)
        
        stock_to_sector = torch.tensor(stock_to_sector_mapping, dtype=torch.long)
        
        # Create fully connected sector graph
        sector_count = len(sector_list)
        sector_edge_index = []
        for i in range(sector_count):
            for j in range(sector_count):
                if i != j:
                    sector_edge_index.append([i, j])
        
        if sector_edge_index:
            sector_edge_index = torch.tensor(sector_edge_index, dtype=torch.long).t().contiguous()
        else:
            sector_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Attach to data object
        data.stock_to_sector = stock_to_sector
        data.sector_edge_index = sector_edge_index
        
        # Store sector metadata
        metadata['sector_to_idx'] = sector_to_idx
        metadata['idx_to_sector'] = {v: k for k, v in sector_to_idx.items()}
        metadata['num_sectors'] = sector_count
        
        print(f"\n✅ Sector-aware graph:")
        print(f"  - {sector_count} sectors")
        print(f"  - stock_to_sector shape: {stock_to_sector.shape}")
        print(f"  - sector_edge_index shape: {sector_edge_index.shape}")

        
        return data, metadata
    
    
    def create_temporal_splits(self, data: Data, metadata: Dict, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[Data, Data, Data]:
        """Create splits"""
        
        num_nodes = data.num_nodes
        num_train = int(num_nodes * train_ratio)
        num_val = int(num_nodes * val_ratio)
        
        torch.manual_seed(42)
        perm = torch.randperm(num_nodes)
        
        train_idx = perm[:num_train]
        val_idx = perm[num_train:num_train + num_val]
        test_idx = perm[num_train + num_val:]
        
        train_data = self._create_subgraph(data, train_idx)
        val_data = self._create_subgraph(data, val_idx)
        test_data = self._create_subgraph(data, test_idx)
        
        print(f"\n✅ Data splits:")
        print(f"   Train: {train_data.num_nodes} stocks")
        print(f"   Val:   {val_data.num_nodes} stocks")
        print(f"   Test:  {test_data.num_nodes} stocks")
        
        return train_data, val_data, test_data
    
    def _create_subgraph(self, data: Data, node_idx: torch.Tensor) -> Data:
        """Create subgraph with sector-aware attributes"""
        x_sub = data.x[node_idx]
        y_sub = data.y[node_idx]
        
        # Filter stock-level edges
        mask = torch.isin(data.edge_index[0], node_idx) & torch.isin(data.edge_index[1], node_idx)
        edge_index_sub = data.edge_index[:, mask]
        
        # Remap node indices
        old_to_new = torch.full((data.num_nodes,), -1, dtype=torch.long)
        old_to_new[node_idx] = torch.arange(len(node_idx))
        edge_index_sub = old_to_new[edge_index_sub]
        
        # Handle sector mapping
        stock_to_sector_sub = data.stock_to_sector[node_idx] if hasattr(data, 'stock_to_sector') else torch.zeros(len(node_idx), dtype=torch.long)
        
        # Sector edges remain the same (all sectors are always connected)
        sector_edge_index_sub = data.sector_edge_index if hasattr(data, 'sector_edge_index') else torch.empty((2, 0), dtype=torch.long)
        
        subgraph = Data(
            x=x_sub, 
            edge_index=edge_index_sub, 
            y=y_sub, 
            num_nodes=len(node_idx)
        )
        
        # Attach sector attributes
        subgraph.stock_to_sector = stock_to_sector_sub
        subgraph.sector_edge_index = sector_edge_index_sub
        
        return subgraph
    
    def _create_targets(self, features: pd.DataFrame, tickers: List[str]) -> torch.Tensor:
        """
        ✅ LEAK-FREE target creation
        Target: Days -5 to -1 (future)
        Volatility: Days -40 to -11 (past, no overlap)
        """
        
        all_future_returns = []
        ticker_return_map = {}
        
        for ticker in tickers:
            return_col = f'{ticker}_returns'
            if return_col in features.columns:
                returns = features[return_col].dropna()
                if len(returns) > 60:
                    # ✅ Target: Days -5 to -1
                    future_return = returns.iloc[-5:].mean()
                    all_future_returns.append(future_return)
                    ticker_return_map[ticker] = future_return
        
        global_median = np.median(all_future_returns) if all_future_returns else 0.0
        
        print("\n🚀 LEAK-FREE Target Creation:")
        print("=" * 60)
        print(f"Global median: {global_median:.6f}")
        print(f"Target window: Days -5 to -1")
        print(f"Feature window: Days -60 to -11")
        print(f"Buffer: 5 days (no overlap)")
        
        targets = []
        up_count = 0
        down_count = 0
        
        for ticker in tickers:
            return_col = f'{ticker}_returns'
            
            if return_col in features.columns:
                returns = features[return_col].dropna()
                
                if len(returns) > 60:
                    try:
                        future_return = ticker_return_map.get(ticker, 0.0)
                        direction = 1 if future_return > global_median else 0
                        
                        if direction == 1:
                            up_count += 1
                        else:
                            down_count += 1
                        
                        # ✅ FIXED: Volatility from Days -60 to -11 (no overlap with target)
                        vol = returns.iloc[-60:-11].std()
                        vol_adj_return = future_return / (vol + 1e-8) if not pd.isna(vol) else 0.0
                        
                        targets.append([future_return, direction, vol_adj_return])
                        
                    except (IndexError, KeyError):
                        targets.append([0.0, 0, 0.0])
                else:
                    targets.append([0.0, 0, 0.0])
            else:
                targets.append([0.0, 0, 0.0])
        
        total = up_count + down_count
        up_ratio = up_count / total * 100 if total > 0 else 0
        
        print(f"\n✅ Target Balance:")
        print(f"   UP:   {up_count} ({up_ratio:.1f}%)")
        print(f"   DOWN: {down_count} ({100-up_ratio:.1f}%)")
        
        if 45 <= up_ratio <= 55:
            print(f"   ✅ PERFECTLY BALANCED!")
        elif 40 <= up_ratio <= 60:
            print(f"   ✅ GOOD BALANCE")
        else:
            print(f"   ⚠️ WARNING: Imbalanced ({up_ratio:.1f}% UP)")
        
        print("=" * 60)
        
        return torch.tensor(targets, dtype=torch.float)
    
    def _get_sector_distribution(self, sectors: Dict[str, str]) -> Dict[str, int]:
        distribution = {}
        for sector in sectors.values():
            distribution[sector] = distribution.get(sector, 0) + 1
        return distribution


def main():
    """Example usage for Indian market data"""
    csv_folder = "indian_data"
    dataset = FinancialDataset(csv_folder, max_stocks=550)
    
    try:
        graph_data, metadata = dataset.prepare_dataset()
        train_data, val_data, test_data = dataset.create_temporal_splits(graph_data, metadata)
        
        print("\n" + "=" * 60)
        print("✅ LEAK-FREE DATA READY FOR TRAINING")
        print("Expected: Honest 52-60% Accuracy")
        print("=" * 60)
        
        return graph_data, metadata, train_data, val_data, test_data
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


if __name__ == "__main__":
    main()
