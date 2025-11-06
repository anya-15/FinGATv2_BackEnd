"""
Enhanced Lightning Module with Hierarchical Stock + Sector Architecture
- Stock-level GAT (intra-sector relationships)
- Sector-level GAT (inter-sector relationships)  
- LSTM temporal encoding
- Learnable loss balancing
- Focal Loss + Balance Penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torchmetrics import Accuracy, MeanSquaredError, MeanAbsoluteError, R2Score
from typing import Dict, Tuple, List, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FOCAL LOSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FocalLoss(nn.Module):
    """Focal Loss - Forces model to learn harder examples"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER MODULES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TemporalEncoder(nn.Module):
    """LSTM-based temporal encoder"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self.layer_norm(x) if x.size(-1) == self.lstm.hidden_size else x
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        return self.layer_norm(h_n[-1])


class AttentionPooling(nn.Module):
    """Attention-based pooling for stock → sector aggregation"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        attention_scores = self.attention(x)
        attention_weights = torch.zeros_like(attention_scores)
        unique_batches = torch.unique(batch)
        
        for batch_id in unique_batches:
            mask = (batch == batch_id)
            batch_scores = attention_scores[mask]
            attention_weights[mask] = F.softmax(batch_scores, dim=0)
        
        weighted_x = attention_weights * x
        return global_mean_pool(weighted_x, batch)


class ImprovedGATv2Layer(nn.Module):
    """Enhanced GAT layer with residual connections"""
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, 
                 dropout: float = 0.3, use_residual: bool = True):
        super().__init__()
        
        self.total_out_dim = out_dim * num_heads
        self.use_residual = use_residual
        
        self.gatv2 = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True,
            bias=True,
            add_self_loops=True
        )
        
        self.layer_norm = nn.LayerNorm(self.total_out_dim)
        self.dropout = nn.Dropout(dropout)
        
        if self.use_residual and in_dim != self.total_out_dim:
            self.residual_proj = nn.Linear(in_dim, self.total_out_dim, bias=False)
        else:
            self.residual_proj = None
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.gatv2(x, edge_index)
        x = self.dropout(x)
        
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            x = x + identity
        
        x = self.layer_norm(x)
        return F.elu(x)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN MODEL: HIERARCHICAL STOCK + SECTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ModernFinGATWithSector(nn.Module):
    """
    Hierarchical architecture:
    1. Stock-level GAT (intra-sector)
    2. Sector-level GAT (inter-sector)
    3. Fusion for final predictions
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.input_dim = config['model']['input_dim']
        self.hidden_dim = config['model']['hidden_dim']
        self.output_dim = config['model'].get('output_dim', 1)
        self.num_heads = config['model']['num_heads']
        self.num_layers = config['model']['num_layers']
        self.dropout = config['model']['dropout']
        self.use_residual = config['model'].get('use_residual', True)
        self.use_temporal = config['model'].get('use_temporal', False)
        
        # Learnable multi-task loss weights
        self.log_vars = nn.Parameter(torch.zeros(3))
        
        # Input normalization
        self.feature_norm = nn.LayerNorm(self.input_dim)
        
        # Temporal encoder (optional, set to False for now)
        if self.use_temporal:
            self.temporal_encoder = TemporalEncoder(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2,
                dropout=self.dropout
            )
            self.input_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.temporal_encoder = None
            self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # LEVEL 1: STOCK-LEVEL GAT
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        gat_dims = []
        current_dim = self.hidden_dim
        for i in range(self.num_layers):
            out_per_head = self.hidden_dim // self.num_heads
            gat_dims.append((current_dim, out_per_head))
            current_dim = out_per_head * self.num_heads
        
        self.stock_gat_layers = nn.ModuleList([
            ImprovedGATv2Layer(
                in_dim=in_d,
                out_dim=out_d,
                num_heads=self.num_heads,
                dropout=self.dropout,
                use_residual=self.use_residual
            )
            for in_d, out_d in gat_dims
        ])
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # AGGREGATION: STOCK → SECTOR
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        self.attention_pool = AttentionPooling(self.hidden_dim)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # LEVEL 2: SECTOR-LEVEL GAT
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        sector_gat_dims = []
        current_dim = self.hidden_dim
        for i in range(self.num_layers):
            out_per_head = self.hidden_dim // self.num_heads
            sector_gat_dims.append((current_dim, out_per_head))
            current_dim = out_per_head * self.num_heads
        
        self.sector_gat_layers = nn.ModuleList([
            ImprovedGATv2Layer(
                in_dim=in_d,
                out_dim=out_d,
                num_heads=self.num_heads,
                dropout=self.dropout,
                use_residual=self.use_residual
            )
            for in_d, out_d in sector_gat_dims
        ])
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FUSION LAYER
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ELU(),
            nn.Dropout(self.dropout * 0.5)
        )
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # TASK-SPECIFIC FEATURE EXTRACTORS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        self.regression_feature = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        self.classification_feature = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        self.ranking_feature = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PREDICTION HEADS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        self.regression_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout * 0.4),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout * 0.2),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Tanh()
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout * 0.4),
            nn.Linear(self.hidden_dim // 2, 2)
        )
        
        self.ranking_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout * 0.4),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Weight initialization
        self.apply(self._init_weights)
        
        # Initialize loss weights
        with torch.no_grad():
            self.log_vars[0] = 0.0
            self.log_vars[1] = -2.0
            self.log_vars[2] = -1.0
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, 'out_features') and module.out_features == 1:
                nn.init.xavier_uniform_(module.weight)
            else:
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(
        self, 
        x_stock: torch.Tensor, 
        edge_index_stock: torch.Tensor, 
        batch_stock: torch.Tensor,
        x_sector: Optional[torch.Tensor],
        edge_index_sector: torch.Tensor, 
        batch_sector: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Normalize stock features
        x_stock = self.feature_norm(x_stock)
        x_stock = self.input_proj(x_stock)
        x_stock = F.dropout(x_stock, p=self.dropout * 0.5, training=self.training)
        
        # Stock-level GAT
        for gat_layer in self.stock_gat_layers:
            x_stock = gat_layer(x_stock, edge_index_stock)
        
        # Aggregate stocks to sectors
        sector_embeds = self.attention_pool(x_stock, batch_stock)
        
        # Sector-level GAT
        for sector_gat_layer in self.sector_gat_layers:
            sector_embeds = sector_gat_layer(sector_embeds, edge_index_sector)
        
        # Broadcast sector info back to stocks
        sector_embeds_broadcast = sector_embeds[batch_stock]
        
        # Fusion
        combined = torch.cat([x_stock, sector_embeds_broadcast], dim=-1)
        combined_embeds = self.fusion_proj(combined)
        
        # Task-specific features
        reg_features = self.regression_feature(combined_embeds)
        clf_features = self.classification_feature(combined_embeds)
        rank_features = self.ranking_feature(combined_embeds)
        
        # Predictions
        returns_pred = self.regression_head(reg_features).squeeze(-1) * 0.1
        movement_pred = self.classification_head(clf_features)
        ranking_scores = self.ranking_head(rank_features).squeeze(-1)
        
        return returns_pred, movement_pred, ranking_scores, combined_embeds
    
    def compute_balanced_loss(
        self, 
        reg_loss: torch.Tensor, 
        clf_loss: torch.Tensor, 
        rank_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Learnable loss balancing"""
        precision_reg = torch.exp(-self.log_vars[0])
        precision_clf = torch.exp(-self.log_vars[1])
        precision_rank = torch.exp(-self.log_vars[2])
        
        weighted_reg = precision_reg * reg_loss + self.log_vars[0]
        weighted_clf = precision_clf * clf_loss + self.log_vars[1]
        weighted_rank = precision_rank * rank_loss + self.log_vars[2]
        
        total_loss = weighted_reg + weighted_clf + weighted_rank
        
        return total_loss, {
            'precision_reg': precision_reg.item(),
            'precision_clf': precision_clf.item(),
            'precision_rank': precision_rank.item()
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA MODULE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FinGATDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict, train_data: Data, val_data: Data, 
                 test_data: Data, metadata: Dict):
        super().__init__()
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.metadata = metadata
        
    def train_dataloader(self):
        return DataLoader([self.train_data], batch_size=1, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader([self.val_data], batch_size=1, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader([self.test_data], batch_size=1, shuffle=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LIGHTNING MODULE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FinGATLightningModule(pl.LightningModule):
    """Enhanced Lightning module with hierarchical architecture"""
    
    def __init__(self, config: Dict, metadata: Dict):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.metadata = metadata
        
        # ✅✅ Use ModernFinGATWithSector
        self.model = ModernFinGATWithSector(config)
        
        # Focal Loss
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
        self._init_metrics()
        self.validation_outputs = []
        self.test_outputs = []
    
    def _init_metrics(self):
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        self.test_accuracy = Accuracy(task='binary')
        
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()
        
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()
    
    def forward(self, x, edge_index, stock_to_sector, sector_edge_index):
        """✅✅ Updated forward with sector information"""
        return self.model(
            x_stock=x,
            edge_index_stock=edge_index,
            batch_stock=stock_to_sector,
            x_sector=None,
            edge_index_sector=sector_edge_index
        )
    
    def create_fully_connected_edges(self, num_nodes, device):
        """Create fully connected edge index for sectors"""
        edge_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_list.append([i, j])
        
        if len(edge_list) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()
        return edge_index
    
    def _shared_step(self, batch, batch_idx: int, stage: str):
        x, edge_index, y = batch.x, batch.edge_index, batch.y
        
        # ✅✅ Get sector assignments
        stock_to_sector = batch.stock_to_sector if hasattr(batch, 'stock_to_sector') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Create sector edges (fully connected)
        num_sectors = stock_to_sector.max().item() + 1 if stock_to_sector.numel() > 0 else 1
        sector_edge_index = self.create_fully_connected_edges(num_sectors, x.device)
        
        # Forward pass
        returns_pred, movement_pred, ranking_scores, embeddings = self.forward(
            x, edge_index, stock_to_sector, sector_edge_index
        )
        
        returns_target = y[:, 0]
        movement_target = y[:, 1].long()
        
        # Focal Loss + Balance Penalty
        classification_loss = self.focal_loss(movement_pred, movement_target)
        
        movement_predictions = torch.argmax(movement_pred, dim=1)
        pred_up_ratio = (movement_predictions == 1).float().mean()
        balance_penalty = torch.abs(pred_up_ratio - 0.5) * 0.5
        
        classification_loss = classification_loss + balance_penalty
        
        # Other losses
        regression_loss = F.mse_loss(returns_pred.squeeze(), returns_target)
        ranking_loss = self._pairwise_ranking_loss(ranking_scores.squeeze(), returns_target)
        
        total_loss, loss_dict = self.model.compute_balanced_loss(
            regression_loss, classification_loss, ranking_loss
        )
        
        # [Rest of logging code stays exactly the same as your original...]
        
        if stage == 'train':
            self.train_accuracy(movement_predictions, movement_target)
            self.train_mse(returns_pred.squeeze(), returns_target)
            self.train_mae(returns_pred.squeeze(), returns_target)
            self.train_r2(returns_pred.squeeze(), returns_target)
            
            self.log(f'{stage}_loss', total_loss, on_epoch=True, prog_bar=True)
            self.log(f'{stage}_regression_loss', regression_loss, on_epoch=True)
            self.log(f'{stage}_classification_loss', classification_loss, on_epoch=True)
            self.log(f'{stage}_ranking_loss', ranking_loss, on_epoch=True)
            self.log(f'{stage}_accuracy', self.train_accuracy, on_epoch=True, prog_bar=True)
            self.log(f'{stage}_mse', self.train_mse, on_epoch=True)
            self.log(f'{stage}_mae', self.train_mae, on_epoch=True)
            self.log(f'{stage}_r2', self.train_r2, on_epoch=True)
            
            self.log('train_up_ratio', pred_up_ratio, on_epoch=True, prog_bar=True)
            self.log('train_balance_penalty', balance_penalty, on_epoch=True)
            
            self.log('reg_weight', loss_dict['precision_reg'], on_epoch=True)
            self.log('clf_weight', loss_dict['precision_clf'], on_epoch=True)
            self.log('rank_weight', loss_dict['precision_rank'], on_epoch=True)
            
        elif stage == 'val':
            self.val_accuracy(movement_predictions, movement_target)
            self.val_mse(returns_pred.squeeze(), returns_target)
            self.val_mae(returns_pred.squeeze(), returns_target)
            self.val_r2(returns_pred.squeeze(), returns_target)
            
            self.log(f'{stage}_loss', total_loss, on_epoch=True, prog_bar=True)
            self.log(f'{stage}_regression_loss', regression_loss, on_epoch=True)
            self.log(f'{stage}_classification_loss', classification_loss, on_epoch=True)
            self.log(f'{stage}_ranking_loss', ranking_loss, on_epoch=True)
            self.log(f'{stage}_accuracy', self.val_accuracy, on_epoch=True, prog_bar=True)
            self.log(f'{stage}_mse', self.val_mse, on_epoch=True)
            self.log(f'{stage}_mae', self.val_mae, on_epoch=True)
            self.log(f'{stage}_r2', self.val_r2, on_epoch=True)
            
            self.log('val_up_ratio', pred_up_ratio, on_epoch=True, prog_bar=True)
            
            self.validation_outputs.append({
                'returns_pred': returns_pred.detach().cpu(),
                'movement_pred': movement_predictions.detach().cpu(),
                'ranking_scores': ranking_scores.detach().cpu(),
                'returns_target': returns_target.detach().cpu(),
                'movement_target': movement_target.detach().cpu()
            })
            
        elif stage == 'test':
            self.test_accuracy(movement_predictions, movement_target)
            self.test_mse(returns_pred.squeeze(), returns_target)
            self.test_mae(returns_pred.squeeze(), returns_target)
            self.test_r2(returns_pred.squeeze(), returns_target)
            
            self.log(f'{stage}_loss', total_loss, on_epoch=True)
            self.log(f'{stage}_regression_loss', regression_loss, on_epoch=True)
            self.log(f'{stage}_classification_loss', classification_loss, on_epoch=True)
            self.log(f'{stage}_ranking_loss', ranking_loss, on_epoch=True)
            self.log(f'{stage}_accuracy', self.test_accuracy, on_epoch=True)
            self.log(f'{stage}_mse', self.test_mse, on_epoch=True)
            self.log(f'{stage}_mae', self.test_mae, on_epoch=True)
            self.log(f'{stage}_r2', self.test_r2, on_epoch=True)
            
            self.log('test_up_ratio', pred_up_ratio, on_epoch=True)
            
            self.test_outputs.append({
                'returns_pred': returns_pred.detach().cpu(),
                'movement_pred': movement_predictions.detach().cpu(),
                'ranking_scores': ranking_scores.detach().cpu(),
                'returns_target': returns_target.detach().cpu(),
                'movement_target': movement_target.detach().cpu()
            })
        
        return total_loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'test')
    
    def on_validation_epoch_end(self):
        if self.validation_outputs:
            self._calculate_and_log_topk_metrics(self.validation_outputs, 'val')
            self.validation_outputs.clear()
    
    def on_test_epoch_end(self):
        if self.test_outputs:
            self._calculate_and_log_topk_metrics(self.test_outputs, 'test')
            self._print_top_k_recommendations()
    
    def _calculate_and_log_topk_metrics(self, outputs: List[Dict], stage: str):
        all_ranking_scores = torch.cat([out['ranking_scores'].squeeze() for out in outputs])
        all_returns_target = torch.cat([out['returns_target'] for out in outputs])
        all_movement_target = torch.cat([out['movement_target'] for out in outputs])
        
        for k in [5, 10, 20]:
            k_actual = min(k, len(all_ranking_scores))
            if k_actual > 0:
                top_k_indices = torch.topk(all_ranking_scores, k=k_actual).indices
                top_k_movements = all_movement_target[top_k_indices]
                precision_at_k = (top_k_movements == 1).float().mean()
                top_k_returns = all_returns_target[top_k_indices]
                avg_return_top_k = top_k_returns.mean()
                
                self.log(f'{stage}_precision@{k}', precision_at_k, on_epoch=True)
                self.log(f'{stage}_avg_return@{k}', avg_return_top_k, on_epoch=True)
        
        mrr = self._calculate_mrr(all_ranking_scores, all_returns_target)
        self.log(f'{stage}_mrr', mrr, on_epoch=True, prog_bar=True)
    
    def _print_top_k_recommendations(self):
        if not self.test_outputs:
            return
            
        all_ranking_scores = torch.cat([out['ranking_scores'].squeeze() for out in self.test_outputs])
        all_returns_pred = torch.cat([out['returns_pred'].squeeze() for out in self.test_outputs])
        all_movement_pred = torch.cat([out['movement_pred'] for out in self.test_outputs])
        all_returns_target = torch.cat([out['returns_target'] for out in self.test_outputs])
        all_movement_target = torch.cat([out['movement_target'] for out in self.test_outputs])
        
        tickers = list(self.metadata['ticker_to_idx'].keys())
        
        print("\n" + "="*80)
        print("🏆 TOP-K STOCK RECOMMENDATIONS")
        print("="*80)
        
        up_count = (all_movement_pred == 1).sum().item()
        down_count = (all_movement_pred == 0).sum().item()
        print(f"\n📊 Prediction Balance:")
        print(f"   UP: {up_count} ({up_count/(up_count+down_count)*100:.1f}%)")
        print(f"   DOWN: {down_count} ({down_count/(up_count+down_count)*100:.1f}%)")
        
        for k in [5, 10]:
            k_actual = min(k, len(all_ranking_scores))
            if k_actual > 0:
                print(f"\n📊 **TOP-{k} RECOMMENDED STOCKS:**")
                print("-"*70)
                
                top_k_indices = torch.topk(all_ranking_scores, k=k_actual).indices
                
                print(f"{ 'Rank':<4} {'Ticker':<8} {'Pred Dir':<8} {'Act Dir':<8} {'Pred Ret':<10} {'Act Ret':<10} {'Score':<8} {'✓'}")
                print("-"*70)
                
                for i, idx in enumerate(top_k_indices):
                    ticker = tickers[idx] if idx < len(tickers) else f"Stock_{idx}"
                    pred_direction = "UP" if all_movement_pred[idx] == 1 else "DOWN"
                    pred_return = f"{all_returns_pred[idx].item():.3f}"
                    ranking_score = f"{all_ranking_scores[idx].item():.3f}"
                    actual_direction = "UP" if all_movement_target[idx] == 1 else "DOWN"
                    actual_return = f"{all_returns_target[idx].item():.3f}"
                    dir_correct = "✅" if pred_direction == actual_direction else "❌"
                    
                    print(f"{i+1:<4} {ticker:<8} {pred_direction:<8} {actual_direction:<8} {pred_return:<10} {actual_return:<10} {ranking_score:<8} {dir_correct}")
    
    def _pairwise_ranking_loss(self, scores, targets):
        if len(scores) < 2:
            return torch.tensor(0.0, device=scores.device)
            
        score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
        loss = torch.clamp(0.1 - score_diff * torch.sign(target_diff), min=0)
        mask = (target_diff != 0).float()
        
        if mask.sum() > 0:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = torch.tensor(0.0, device=scores.device)
        
        return loss
    
    def _calculate_mrr(self, scores, targets):
        if len(scores) == 0:
            return torch.tensor(0.0)
            
        sorted_indices = torch.argsort(scores, descending=True)
        best_stock_idx = torch.argmax(targets)
        rank_of_best = (sorted_indices == best_stock_idx).nonzero(as_tuple=True)[0]
        
        if len(rank_of_best) > 0:
            return 1.0 / (rank_of_best[0].float() + 1)
        else:
            return torch.tensor(0.0)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=5e-4,
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',
            factor=0.5, 
            patience=10
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_mrr'
            }
        }
