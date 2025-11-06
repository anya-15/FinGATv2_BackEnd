"""
🎯 MULTI-SCALE STOCK PREDICTOR WITH CONFIDENCE & SECTOR FILTERING
- Loads RL-selected features + best checkpoint from latest manifest
- Works with multi-scale temporal features (73 → masked to N by RL)
- Produces Top-5/10/20 UP-only picks with confidence filtering
"""

import os
import sys
import json
import glob
import warnings
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Use project-style imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.data_loader import FinancialDataset

LATEST_MANIFEST = "rl_models/selected_runs/latest_manifest.json"
OUTPUT_DIR = "predictions"
CONF_MIN = 0.40          # probability threshold on UP (0..1)
TOPK_LIST = [5, 10, 20]  # which Top-K files/prints to produce


# ───────────────────────────────────────────────────────────────
# MODEL DEFINITION (kept compatible with your checkpoint)
# ───────────────────────────────────────────────────────────────


class ImprovedFinGAT(nn.Module):
    """FinGAT model - works with any input dimension"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # Learnable log-variances (kept from original design)
        self.log_vars = nn.Parameter(torch.zeros(3))

        self.feature_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            concat = (i < num_layers - 1)
            out_heads = num_heads if concat else 1

            self.gat_layers.append(
                GATv2Conv(in_dim, hidden_dim, heads=out_heads,
                          dropout=dropout, concat=concat, add_self_loops=True)
            )

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * num_heads if i < num_layers - 1 else hidden_dim)
            for i in range(num_layers)
        ])

        # task-specific feature transforms
        self.regression_feature = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classification_feature = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.ranking_feature = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(max(8, hidden_dim // 2), max(4, hidden_dim // 4)),
            nn.ReLU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(max(4, hidden_dim // 4), 1),
            nn.Tanh()
        )

        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(max(8, hidden_dim // 2), 2)
        )

        self.ranking_head = nn.Sequential(
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(max(8, hidden_dim // 2), 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        x = self.feature_norm(x)
        x = self.input_projection(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout * 0.5, training=self.training)

        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x = gat_layer(x, edge_index)
            x = layer_norm(x)
            x = F.elu(x)
            if i < len(self.gat_layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        embeddings = x

        reg_features = self.regression_feature(embeddings)
        clf_features = self.classification_feature(embeddings)
        rank_features = self.ranking_feature(embeddings)

        returns_pred = self.regression_head(reg_features) * 0.1
        movement_pred = self.classification_head(clf_features)
        ranking_scores = self.ranking_head(rank_features)

        return returns_pred, movement_pred, ranking_scores, embeddings


# ───────────────────────────────────────────────────────────────
# FILTERING / DIVERSIFICATION
# ───────────────────────────────────────────────────────────────


def filter_by_confidence(results_df, min_confidence=15):
    original_count = len(results_df)
    filtered = results_df[results_df['Confidence_%'] >= min_confidence].copy()
    filtered = filtered.reset_index(drop=True)
    filtered['Rank'] = range(1, len(filtered) + 1)

    print(f"\n🎯 CONFIDENCE FILTERING:")
    print(f"   Original predictions: {original_count}")
    print(f"   High confidence (>{min_confidence}%): {len(filtered)}")
    print(f"   Filtered out: {original_count - len(filtered)} low-confidence picks")

    return filtered


def diversify_by_sector(results_df, top_k=20, max_per_sector=3):
    selected = []
    sector_counts = {}

    for _, row in results_df.iterrows():
        sector = row['Sector']
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        selected.append(row.to_dict())
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected) >= top_k:
            break

    diversified = pd.DataFrame(selected)
    if len(diversified) > 0:
        diversified['Rank'] = range(1, len(diversified) + 1)

        print(f"\n📊 SECTOR DIVERSIFICATION:")
        print(f"   Selected stocks: {len(diversified)}")
        print(f"   Max per sector: {max_per_sector}")
        print(f"   Sectors covered: {diversified['Sector'].nunique()}")

        sector_dist = diversified['Sector'].value_counts()
        for sector, count in sector_dist.items():
            print(f"   • {sector}: {count} stocks")
    else:
        print("\n📊 SECTOR DIVERSIFICATION:")
        print("   No stocks available after filtering.")

    return diversified


# ───────────────────────────────────────────────────────────────
# MANIFEST / LOADING HELPERS
# ───────────────────────────────────────────────────────────────


def load_manifest(path=LATEST_MANIFEST):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Latest manifest not found: {path}")
    with open(path, "r") as f:
        m = json.load(f)
    for key in ["features_path", "checkpoint_path"]:
        if key not in m or not m[key]:
            raise ValueError(f"Manifest missing required key: {key}")
    return m


def load_feature_indices(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature mask not found: {path}")
    mask = np.load(path)
    idx = np.where(mask == 1)[0]
    if len(idx) == 0:
        raise ValueError("Loaded feature mask has zero selected features.")
    print(f"✅ Using RL-selected features: {len(idx)} → head: {idx[:10].tolist()}")
    return idx


def apply_feature_mask_to_graph(data, selected_idx):
    data = data.clone()
    data.x = data.x[:, selected_idx]
    return data


# ───────────────────────────────────────────────────────────────
# PREDICTOR
# ───────────────────────────────────────────────────────────────


class TomorrowPredictor:
    """Predicts tomorrow's best stocks using trained model with RL-selected features"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("\n" + "="*80)
        print("🚀 MULTI-SCALE STOCK PREDICTOR - INITIALIZING")
        print("="*80)

        # 1) Load manifest → features + checkpoint (+ optional hparams)
        manifest = load_manifest()
        self.features_path = manifest["features_path"]
        self.checkpoint_path = manifest["checkpoint_path"]
        self.hparams_path = manifest.get("hparams_path")

        print(f"\n📄 Using manifest:")
        print(f"   features_path   : {self.features_path}")
        print(f"   checkpoint_path : {self.checkpoint_path}")
        if self.hparams_path and os.path.exists(self.hparams_path):
            print(f"   hparams_path    : {self.hparams_path}")

        # 2) Dataset → graph
        print(f"\n📊 Loading stock data from: {data_path}")
        self.dataset = FinancialDataset(
            csv_folder_path=data_path,
            max_stocks=550
        )
        full_graph, self.metadata = self.dataset.prepare_dataset()

        # 3) Apply feature mask
        self.selected_idx = load_feature_indices(self.features_path)
        masked_graph = apply_feature_mask_to_graph(full_graph, self.selected_idx)
        self.graph_data = masked_graph

        # Update metadata with masked dimension
        self.metadata = dict(self.metadata)
        self.metadata['num_features'] = int(len(self.selected_idx))

        print(f"\n✅ Data ready:")
        print(f"   Masked features: {self.metadata['num_features']}")
        print(f"   Stocks: {self.metadata.get('num_stocks', 'unknown')}")
        print(f"   Edges: {self.metadata.get('num_edges', 'unknown')}")

        self.model = self._load_model()

        print("\n✅ Predictor ready!")
        print("="*80)

    def _load_model(self):
        """Load trained model matching the masked input dimension"""
        print(f"\n🤖 Loading model from: {self.checkpoint_path}")

        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Derive hparams if provided
        hidden_dim = 128
        dropout = 0.3
        if self.hparams_path and os.path.exists(self.hparams_path):
            try:
                with open(self.hparams_path, "r") as f:
                    h = json.load(f)
                hidden_dim = int(h.get("hidden_dim", hidden_dim))
                dropout = float(h.get("dropout", dropout))
            except Exception:
                pass

        input_dim = self.metadata['num_features']
        model = ImprovedFinGAT(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            dropout=dropout
        )

        # Handle Lightning-style state dicts
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('model.', '') if key.startswith('model.') else key
            new_state_dict[new_key] = value

        # Load with strict=False to allow small mismatches
        model.load_state_dict(new_state_dict, strict=False)
        model.to(self.device)
        model.eval()

        print(f"✅ Model loaded with input_dim={input_dim}, hidden_dim={hidden_dim}, dropout={dropout}")
        return model

    @torch.no_grad()
    def predict_all_stocks(self):
        """Generate predictions for all stocks"""
        print("\n" + "="*80)
        print(f"📈 GENERATING PREDICTIONS FOR TOMORROW")
        print(f"   Date: {datetime.now().strftime('%A, %B %d, %Y - %I:%M %p')}")
        print("="*80)

        print("\n🔮 Running model inference...")

        x = self.graph_data.x.to(self.device)
        edge_index = self.graph_data.edge_index.to(self.device)

        returns_pred, movement_logits, ranking_scores, _ = self.model(x, edge_index)

        movement_probs = torch.softmax(movement_logits, dim=1)
        prob_up = movement_probs[:, 1].cpu().numpy()

        # Confidence is the distance from 0.5 scaled as a percentage
        confidence_pct = np.abs(prob_up - 0.5) * 200.0
        pred_dir = (prob_up >= 0.5).astype(int)

        print(f"\n📊 UP probability range: {prob_up.min():.3f} - {prob_up.max():.3f}")

        returns_pred = returns_pred.cpu().numpy().flatten()
        ranking_scores = ranking_scores.cpu().numpy().flatten()

        idx_to_ticker = self.metadata.get('idx_to_ticker', {})
        tickers = [idx_to_ticker.get(i, f"T{i}") for i in range(x.size(0))]
        sectors_map = self.metadata.get('sectors', {})
        sectors = [sectors_map.get(t, 'Other') for t in tickers]

        results_df = pd.DataFrame({
            'Ticker': tickers[:len(prob_up)],
            'Sector': sectors[:len(prob_up)],
            'Direction': ['UP' if d == 1 else 'DOWN' for d in pred_dir],
            'Confidence_%': confidence_pct[:len(prob_up)],
            'Expected_Return_%': (returns_pred[:len(prob_up)] * 100.0),
            'Ranking_Score': ranking_scores[:len(prob_up)],
            'UP_Probability': prob_up[:len(prob_up)],
        })

        # UP-only with probability threshold
        up_only = results_df[(results_df['Direction'] == 'UP') & (results_df['UP_Probability'] >= CONF_MIN)].copy()

        # Sort by ranking score
        up_only = up_only.sort_values('Ranking_Score', ascending=False).reset_index(drop=True)
        up_only['Rank'] = range(1, len(up_only) + 1)

        total = len(results_df)
        up_count = (results_df['Direction'] == 'UP').sum()
        down_count = total - up_count

        print(f"\n✅ Prediction Balance (before filtering):")
        print(f"   UP: {up_count} ({up_count/total*100:.1f}%)")
        print(f"   DOWN: {down_count} ({down_count/total*100:.1f}%)")
        print(f"   After UP+confidence filter: {len(up_only)} picks (Prob_UP ≥ {CONF_MIN:.2f})")

        return results_df, up_only

    def save_outputs(self, full_df, up_df):
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Save full and filtered
        full_path = os.path.join(OUTPUT_DIR, f"predictions_{ts}.csv")
        up_path = os.path.join(OUTPUT_DIR, f"predictions_up_only_{ts}.csv")
        full_df.to_csv(full_path, index=False)
        up_df.to_csv(up_path, index=False)

        # Save Top-K
        paths = {}
        for k in TOPK_LIST:
            topk = up_df.head(k)
            p = os.path.join(OUTPUT_DIR, f"top{k}_{ts}.csv")
            topk.to_csv(p, index=False)
            paths[f"top{k}"] = p

        print(f"\n💾 Saved:")
        print(f"   Full: {full_path}")
        print(f"   UP-only: {up_path}")
        for k, p in paths.items():
            print(f"   {k.upper()}: {p}")

        return full_path, up_path, paths

    def print_topk(self, up_df):
        for k in TOPK_LIST:
            topk = up_df.head(k)
            print(f"\n=== TOP-{k} (UP-only, Prob_UP ≥ {CONF_MIN:.2f}) ===")
            if topk.empty:
                print("No picks passed the filter.")
                continue
            cols = ["Rank", "Ticker", "Sector", "UP_Probability", "Confidence_%", "Expected_Return_%", "Ranking_Score"]
            view = topk[cols].copy()
            for c in ["UP_Probability", "Confidence_%", "Expected_Return_%", "Ranking_Score"]:
                view[c] = view[c].map(lambda x: f"{x:.3f}")
            print(view.to_string(index=False))


# ───────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────


def main():
    try:
        data_path = 'indian_data'
        predictor = TomorrowPredictor(data_path)

        full_df, up_df = predictor.predict_all_stocks()

        # Optional: sector diversification on UP picks
        # up_df = diversify_by_sector(up_df, top_k=20, max_per_sector=3)

        predictor.print_topk(up_df)
        predictor.save_outputs(full_df, up_df)

        print("\n" + "="*80)
        print("✅ PREDICTION COMPLETE!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 TIP: Ensure latest_manifest.json exists and paths are valid.")


if __name__ == "__main__":
    main()
import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, 'data')
from data_loader import FinancialDataset  # from your data/ folder

LATEST_MANIFEST = "rl_models/selected_runs/latest_manifest.json"
OUTPUT_DIR = "predictions"
CONF_MIN = 0.40         
TOPK_LIST = [5, 10, 20] 

class ImprovedFinGAT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.log_vars = nn.Parameter(torch.zeros(3))
        self.feature_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            concat = (i < num_layers - 1)
            out_heads = num_heads if concat else 1
            self.gat_layers.append(
                GATv2Conv(in_dim, hidden_dim, heads=out_heads,
                          dropout=dropout, concat=concat, add_self_loops=True)
            )
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * num_heads if i < num_layers - 1 else hidden_dim)
            for i in range(num_layers)
        ])
        self.regression_feature = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classification_feature = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.ranking_feature = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(hidden_dim // 4, 1),
            nn.Tanh()
        )
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 2, 2)
        )
        self.ranking_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        x = self.feature_norm(x)
        x = self.input_projection(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout * 0.5, training=self.training)

        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x = gat_layer(x, edge_index)
            x = layer_norm(x)
            x = F.elu(x)
            if i < len(self.gat_layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        embeddings = x
        reg_features = self.regression_feature(embeddings)
        clf_features = self.classification_feature(embeddings)
        rank_features = self.ranking_feature(embeddings)
        returns_pred = self.regression_head(reg_features) * 0.1
        movement_pred = self.classification_head(clf_features)
        ranking_scores = self.ranking_head(rank_features)
        return returns_pred, movement_pred, ranking_scores, embeddings

def filter_by_confidence(results_df, min_confidence=15):
    original_count = len(results_df)
    filtered = results_df[results_df['Confidence_%'] >= min_confidence].copy()
    filtered = filtered.reset_index(drop=True)
    filtered['Rank'] = range(1, len(filtered) + 1)
    print(f"\n🎯 CONFIDENCE FILTERING:")
    print(f"   Original predictions: {original_count}")
    print(f"   High confidence (>{min_confidence}%): {len(filtered)}")
    print(f"   Filtered out: {original_count - len(filtered)} low-confidence picks")
    return filtered

def diversify_by_sector(results_df, top_k=20, max_per_sector=3):
    selected = []
    sector_counts = {}
    for _, row in results_df.iterrows():
        sector = row['Sector']
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        selected.append(row.to_dict())
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected) >= top_k:
            break
    diversified = pd.DataFrame(selected)
    if len(diversified) > 0:
        diversified['Rank'] = range(1, len(diversified) + 1)
        print(f"\n📊 SECTOR DIVERSIFICATION:")
        print(f"   Selected stocks: {len(diversified)}")
        print(f"   Max per sector: {max_per_sector}")
        print(f"   Sectors covered: {diversified['Sector'].nunique()}")
        sector_dist = diversified['Sector'].value_counts()
        for sector, count in sector_dist.items():
            print(f"   • {sector}: {count} stocks")
    else:
        print("\n📊 SECTOR DIVERSIFICATION:")
        print("   No stocks available after filtering.")
    return diversified

def load_manifest(path=LATEST_MANIFEST):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Latest manifest not found: {path}")
    with open(path, "r") as f:
        m = json.load(f)
    for key in ["features_path", "checkpoint_path"]:
        if key not in m:
            raise ValueError(f"Manifest missing required key: {key}")
    return m

def load_feature_indices(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature mask not found: {path}")
    mask = np.load(path)
    idx = np.where(mask == 1)[0]
    if len(idx) == 0:
        raise ValueError("Loaded feature mask has zero selected features.")
    print(f"✅ Using RL-selected features: {len(idx)} → head: {idx[:10].tolist()}")
    return idx

def apply_feature_mask_to_graph(data, selected_idx):
    data = data.clone()
    data.x = data.x[:, selected_idx]
    return data

class TomorrowPredictor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("\n" + "="*80)
        print("🚀 MULTI-SCALE STOCK PREDICTOR - INITIALIZING")
        print("="*80)
        manifest = load_manifest()
        self.features_path = manifest["features_path"]
        self.checkpoint_path = manifest["checkpoint_path"]
        self.hparams_path = manifest.get("hparams_path")
        print(f"\n📄 Using manifest:")
        print(f"   features_path   : {self.features_path}")
        print(f"   checkpoint_path : {self.checkpoint_path}")
        if self.hparams_path and os.path.exists(self.hparams_path):
            print(f"   hparams_path    : {self.hparams_path}")
        print(f"\n📊 Loading stock data from: {data_path}")
        self.dataset = FinancialDataset(
            csv_folder_path=data_path,
            max_stocks=550
        )
        full_graph, self.metadata = self.dataset.prepare_dataset()
        self.selected_idx = load_feature_indices(self.features_path)
        masked_graph = apply_feature_mask_to_graph(full_graph, self.selected_idx)
        self.graph_data = masked_graph
        self.metadata = dict(self.metadata)
        self.metadata['num_features'] = int(len(self.selected_idx))
        print(f"\n✅ Data ready:")
        print(f"   Masked features: {self.metadata['num_features']}")
        print(f"   Stocks: {self.metadata['num_stocks']}")
        print(f"   Edges: {self.metadata['num_edges']}")
        self.model = self._load_model()
        print("\n✅ Predictor ready!")
        print("="*80)

    def _load_model(self):
        print(f"\n🤖 Loading model from: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        hidden_dim = 128
        dropout = 0.3
        if self.hparams_path and os.path.exists(self.hparams_path):
            try:
                with open(self.hparams_path, "r") as f:
                    h = json.load(f)
                hidden_dim = int(h.get("hidden_dim", hidden_dim))
                dropout = float(h.get("dropout", dropout))
            except Exception:
                pass
        input_dim = self.metadata['num_features']
        model = ImprovedFinGAT(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            dropout=dropout
        )
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('model.', '') if key.startswith('model.') else key
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict, strict=False)
        model.to(self.device)
        model.eval()
        print(f"✅ Model loaded with input_dim={input_dim}, hidden_dim={hidden_dim}, dropout={dropout}")
        return model

    @torch.no_grad()
    def predict_all_stocks(self):
        print("\n" + "="*80)
        print(f"📈 GENERATING PREDICTIONS FOR TOMORROW")
        print(f"   Date: {datetime.now().strftime('%A, %B %d, %Y - %I:%M %p')}")
        print("="*80)
        print("\n🔮 Running model inference...")
        x = self.graph_data.x.to(self.device)
        edge_index = self.graph_data.edge_index.to(self.device)
        returns_pred, movement_logits, ranking_scores, _ = self.model(x, edge_index)
        movement_probs = torch.softmax(movement_logits, dim=1)
        prob_up = movement_probs[:, 1].cpu().numpy()
        confidence_pct = np.abs(prob_up - 0.5) * 200.0
        pred_dir = (prob_up >= 0.5).astype(int)
        print(f"\n📊 UP probability range: {prob_up.min():.3f} - {prob_up.max():.3f}")
        returns_pred = returns_pred.cpu().numpy().flatten()
        ranking_scores = ranking_scores.cpu().numpy().flatten()
        tickers = list(self.metadata['idx_to_ticker'].values())
        sectors = [self.metadata['sectors'].get(t, 'Other') for t in tickers]
        results_df = pd.DataFrame({
            'Ticker': tickers[:len(prob_up)],
            'Sector': sectors[:len(prob_up)],
            'Direction': ['UP' if d == 1 else 'DOWN' for d in pred_dir],
            'Confidence_%': confidence_pct[:len(prob_up)],
            'Expected_Return_%': (returns_pred[:len(prob_up)] * 100.0),
            'Ranking_Score': ranking_scores[:len(prob_up)],
            'UP_Probability': prob_up[:len(prob_up)],
        })
        up_only = results_df[(results_df['Direction'] == 'UP') & (results_df['UP_Probability'] >= CONF_MIN)].copy()
        up_only = up_only.sort_values('Ranking_Score', ascending=False).reset_index(drop=True)
        up_only['Rank'] = range(1, len(up_only) + 1)
        total = len(results_df)
        up_count = (results_df['Direction'] == 'UP').sum()
        down_count = total - up_count
        print(f"\n✅ Prediction Balance (before filtering):")
        print(f"   UP: {up_count} ({up_count/total*100:.1f}%)")
        print(f"   DOWN: {down_count} ({down_count/total*100:.1f}%)")
        print(f"   After UP+confidence filter: {len(up_only)} picks (Prob_UP ≥ {CONF_MIN:.2f})")
        return results_df, up_only

    def save_outputs(self, full_df, up_df):
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        full_path = os.path.join(OUTPUT_DIR, f"predictions_{ts}.csv")
        up_path = os.path.join(OUTPUT_DIR, f"predictions_up_only_{ts}.csv")
        full_df.to_csv(full_path, index=False)
        up_df.to_csv(up_path, index=False)
        paths = {}
        for k in TOPK_LIST:
            topk = up_df.head(k)
            p = os.path.join(OUTPUT_DIR, f"top{k}_{ts}.csv")
            topk.to_csv(p, index=False)
            paths[f"top{k}"] = p
        print(f"\n💾 Saved:")
        print(f"   Full: {full_path}")
        print(f"   UP-only: {up_path}")
        for k, p in paths.items():
            print(f"   {k.upper()}: {p}")
        return full_path, up_path, paths

    def print_topk(self, up_df):
        for k in TOPK_LIST:
            topk = up_df.head(k)
            print(f"\n=== TOP-{k} (UP-only, Prob_UP ≥ {CONF_MIN:.2f}) ===")
            if topk.empty:
                print("No picks passed the filter.")
                continue
            cols = ["Rank", "Ticker", "Sector", "UP_Probability", "Confidence_%", "Expected_Return_%", "Ranking_Score"]
            view = topk[cols].copy()
            for c in ["UP_Probability", "Confidence_%", "Expected_Return_%", "Ranking_Score"]:
                view[c] = view[c].map(lambda x: f"{x:.3f}")
            print(view.to_string(index=False))

def main():
    try:
        data_path = 'indian_data'
        predictor = TomorrowPredictor(data_path)
        full_df, up_df = predictor.predict_all_stocks()
        # up_df = diversify_by_sector(up_df, top_k=20, max_per_sector=3)
        predictor.print_topk(up_df)
        predictor.save_outputs(full_df, up_df)
        print("\n" + "="*80)
        print("✅ PREDICTION COMPLETE!")
        print("="*80)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 TIP: Ensure latest_manifest.json exists and paths are valid.")

if __name__ == "__main__":
    main()
