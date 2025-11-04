import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from typing import Dict, Optional, Tuple


class TemporalEncoder(nn.Module):
    """LSTM-based temporal encoder for capturing sequential patterns"""
    
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
        temporal_features = h_n[-1]
        return self.layer_norm(temporal_features)


class AttentionPooling(nn.Module):
    """Attention-based pooling for aggregating stocks to sectors"""
    
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
        pooled = global_mean_pool(weighted_x, batch)
        return pooled


class ImprovedGATv2Layer(nn.Module):
    """GAT layer with proper residual connections and regularization"""
    
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


class ModernFinGATWithSector(nn.Module):
    """
    Production-ready FinGAT with:
    - Temporal modeling via LSTM
    - Learnable loss balancing
    - Proper regularization
    - Fixed architecture bugs
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.input_dim = config['model']['input_dim']
        self.hidden_dim = config['model']['hidden_dim']
        self.output_dim = config['model']['output_dim']
        self.num_heads = config['model']['num_heads']
        self.num_layers = config['model']['num_layers']
        self.dropout = config['model']['dropout']
        self.use_residual = config['model'].get('use_residual', True)
        self.use_temporal = config['model'].get('use_temporal', True)
        
        # Learnable multi-task loss weights (log variance parameterization)
        self.log_vars = nn.Parameter(torch.zeros(3))  # [regression, classification, ranking]
        
        # Input normalization
        self.feature_norm = nn.LayerNorm(self.input_dim)
        
        # Temporal encoder (optional)
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
        
        # Calculate dimensions for each GAT layer
        gat_dims = []
        current_dim = self.hidden_dim
        for i in range(self.num_layers):
            out_per_head = self.hidden_dim // self.num_heads
            gat_dims.append((current_dim, out_per_head))
            current_dim = out_per_head * self.num_heads
        
        # Stock-level GAT layers
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
        
        # Attention pooling for sector aggregation
        self.attention_pool = AttentionPooling(self.hidden_dim)
        
        # Sector-level GAT layers
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
        
        # Fusion layer
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ELU(),
            nn.Dropout(self.dropout * 0.5)
        )
        
        # Task-specific feature extractors
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
        
        # Prediction heads with proper scaling
        self.regression_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout * 0.4),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout * 0.2),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Tanh()  # Bound predictions to reasonable range
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
            nn.Sigmoid()  # Normalize ranking scores
        )
        
        # Proper weight initialization
        self.apply(self._init_weights)
        
        # Initialize log_vars to balance initial losses
        with torch.no_grad():
            self.log_vars[0] = 0.0  # regression
            self.log_vars[1] = -2.0  # classification (reduce influence)
            self.log_vars[2] = -1.0  # ranking
    
    def _init_weights(self, module):
        """Xavier/Kaiming initialization based on activation"""
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
    
    def compute_balanced_loss(
        self, 
        reg_loss: torch.Tensor, 
        clf_loss: torch.Tensor, 
        rank_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Uncertainty-based multi-task loss balancing.
        Returns total loss and individual weighted losses.
        """
        precision_reg = torch.exp(-self.log_vars[0])
        precision_clf = torch.exp(-self.log_vars[1])
        precision_rank = torch.exp(-self.log_vars[2])
        
        weighted_reg = precision_reg * reg_loss + self.log_vars[0]
        weighted_clf = precision_clf * clf_loss + self.log_vars[1]
        weighted_rank = precision_rank * rank_loss + self.log_vars[2]
        
        total_loss = weighted_reg + weighted_clf + weighted_rank
        
        loss_dict = {
            'weighted_reg': weighted_reg.item(),
            'weighted_clf': weighted_clf.item(),
            'weighted_rank': weighted_rank.item(),
            'precision_reg': precision_reg.item(),
            'precision_clf': precision_clf.item(),
            'precision_rank': precision_rank.item()
        }
        
        return total_loss, loss_dict
    
    def forward(
        self, 
        x_stock: torch.Tensor, 
        edge_index_stock: torch.Tensor, 
        batch_stock: torch.Tensor,
        x_sector: Optional[torch.Tensor],
        edge_index_sector: torch.Tensor, 
        batch_sector: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Feature normalization
        if x_stock.dim() == 3:
            batch_size, seq_len, features = x_stock.shape
            x_stock = x_stock.view(-1, features)
            x_stock = self.feature_norm(x_stock)
            x_stock = x_stock.view(batch_size, seq_len, features)
            
            if self.temporal_encoder:
                x_stock = self.temporal_encoder(x_stock)
        else:
            x_stock = self.feature_norm(x_stock)
        
        x_stock = self.input_proj(x_stock)
        x_stock = F.dropout(x_stock, p=self.dropout * 0.5, training=self.training)
        
        for gat_layer in self.stock_gat_layers:
            x_stock = gat_layer(x_stock, edge_index_stock)
        
        sector_embeds = self.attention_pool(x_stock, batch_stock)
        
        for sector_gat_layer in self.sector_gat_layers:
            sector_embeds = sector_gat_layer(sector_embeds, edge_index_sector)
        
        sector_embeds_broadcast = sector_embeds[batch_stock]
        
        combined = torch.cat([x_stock, sector_embeds_broadcast], dim=-1)
        combined_embeds = self.fusion_proj(combined)
        
        reg_features = self.regression_feature(combined_embeds)
        clf_features = self.classification_feature(combined_embeds)
        rank_features = self.ranking_feature(combined_embeds)
        
        returns_pred = self.regression_head(reg_features).squeeze(-1) * 0.1
        movement_pred = self.classification_head(clf_features)
        ranking_scores = self.ranking_head(rank_features).squeeze(-1)
        
        return returns_pred, movement_pred, ranking_scores, combined_embeds