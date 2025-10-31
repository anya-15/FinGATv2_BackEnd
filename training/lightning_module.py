# # """
# # Complete FinGAT Lightning Module with Model Architecture - WORKING VERSION
# # """

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import pytorch_lightning as pl
# # from torch_geometric.data import Data
# # from torch_geometric.loader import DataLoader
# # from torch_geometric.nn import GATv2Conv, global_mean_pool
# # from torchmetrics import Accuracy, MeanSquaredError, MeanAbsoluteError, R2Score
# # from typing import Dict, Any, Tuple, List, Optional
# # import numpy as np


# # class ModernFinGAT(nn.Module):
# #     """Modern Financial Graph Attention Network with GATv2"""
    
# #     def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 8, 
# #                  num_layers: int = 3, dropout: float = 0.3):
# #         super(ModernFinGAT, self).__init__()
        
# #         self.input_dim = input_dim
# #         self.hidden_dim = hidden_dim
# #         self.num_heads = num_heads
# #         self.num_layers = num_layers
# #         self.dropout = dropout
        
# #         # Input projection
# #         self.input_projection = nn.Linear(input_dim, hidden_dim)
        
# #         # GAT layers
# #         self.gat_layers = nn.ModuleList()
        
# #         # First layer
# #         self.gat_layers.append(
# #             GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
# #         )
        
# #         # Hidden layers
# #         for _ in range(num_layers - 2):
# #             self.gat_layers.append(
# #                 GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, 
# #                          dropout=dropout, concat=True)
# #             )
        
# #         # Last layer (no concatenation)
# #         self.gat_layers.append(
# #             GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, 
# #                      dropout=dropout, concat=False)
# #         )
        
# #         # Multi-task prediction heads
# #         self.regression_head = nn.Sequential(
# #             nn.Linear(hidden_dim, hidden_dim // 2),
# #             nn.ReLU(),
# #             nn.Dropout(dropout),
# #             nn.Linear(hidden_dim // 2, 1)
# #         )
        
# #         self.classification_head = nn.Sequential(
# #             nn.Linear(hidden_dim, hidden_dim // 2),
# #             nn.ReLU(),
# #             nn.Dropout(dropout),
# #             nn.Linear(hidden_dim // 2, 2)  # Binary classification (Up/Down)
# #         )
        
# #         self.ranking_head = nn.Sequential(
# #             nn.Linear(hidden_dim, hidden_dim // 2),
# #             nn.ReLU(),
# #             nn.Dropout(dropout),
# #             nn.Linear(hidden_dim // 2, 1)  # Ranking score
# #         )
        
# #         # Initialize weights
# #         self._init_weights()
    
# #     def _init_weights(self):
# #         """Initialize model weights"""
# #         for module in self.modules():
# #             if isinstance(module, nn.Linear):
# #                 nn.init.xavier_uniform_(module.weight)
# #                 if module.bias is not None:
# #                     nn.init.zeros_(module.bias)
    
# #     def forward(self, x, edge_index):
# #         """Forward pass through the model"""
        
# #         # Input projection
# #         x = self.input_projection(x)
# #         x = F.relu(x)
# #         x = F.dropout(x, p=self.dropout, training=self.training)
        
# #         # GAT layers
# #         for i, gat_layer in enumerate(self.gat_layers):
# #             x_residual = x if i > 0 else None
# #             x = gat_layer(x, edge_index)
            
# #             if i < len(self.gat_layers) - 1:  # Not the last layer
# #                 x = F.elu(x)
# #                 x = F.dropout(x, p=self.dropout, training=self.training)
                
# #                 # Add residual connection if dimensions match
# #                 if x_residual is not None and x_residual.size(-1) == x.size(-1):
# #                     x = x + x_residual
        
# #         # Final activation
# #         embeddings = F.elu(x)
        
# #         # Multi-task predictions
# #         returns_pred = self.regression_head(embeddings)
# #         movement_pred = self.classification_head(embeddings)
# #         ranking_scores = self.ranking_head(embeddings)
        
# #         return returns_pred, movement_pred, ranking_scores, embeddings


# # class FinGATDataModule(pl.LightningDataModule):
# #     """PyTorch Lightning Data Module for FinGAT"""
    
# #     def __init__(self, config: Dict, data: Data, metadata: Dict, batch_size: int = 1):
# #         super().__init__()
# #         self.config = config
# #         self.data = data
# #         self.metadata = metadata
# #         self.batch_size = batch_size
        
# #     def setup(self, stage: Optional[str] = None):
# #         """Split data into train/val/test"""
# #         if stage == "fit" or stage is None:
# #             self.train_data = self.data
# #             self.val_data = self.data
            
# #         if stage == "test" or stage is None:
# #             self.test_data = self.data
    
# #     def train_dataloader(self):
# #         """Training data loader"""
# #         return DataLoader([self.train_data], batch_size=self.batch_size, shuffle=False)
    
# #     def val_dataloader(self):
# #         """Validation data loader"""  
# #         return DataLoader([self.val_data], batch_size=self.batch_size, shuffle=False)
    
# #     def test_dataloader(self):
# #         """Test data loader"""
# #         return DataLoader([self.test_data], batch_size=self.batch_size, shuffle=False)


# # class FinGATLightningModule(pl.LightningModule):
# #     """Enhanced Lightning module with comprehensive metric tracking"""
    
# #     def __init__(self, config: Dict, metadata: Dict):
# #         super().__init__()
# #         self.save_hyperparameters()
        
# #         self.config = config
# #         self.metadata = metadata
        
# #         # Initialize model
# #         self.model = ModernFinGAT(
# #             input_dim=metadata['num_features'],
# #             hidden_dim=config['model']['hidden_dim'],
# #             num_heads=config['model']['num_heads'],
# #             num_layers=config['model'].get('num_layers', 3),
# #             dropout=config['model'].get('dropout', 0.3)
# #         )
        
# #         # Initialize metrics for tracking
# #         self._init_metrics()
        
# #         # Store predictions for top-K analysis
# #         self.validation_outputs = []
# #         self.test_outputs = []
        
# #     def _init_metrics(self):
# #         """Initialize all tracking metrics"""
        
# #         # Classification metrics (Up/Down prediction)
# #         self.train_accuracy = Accuracy(task='binary')
# #         self.val_accuracy = Accuracy(task='binary')
# #         self.test_accuracy = Accuracy(task='binary')
        
# #         # Regression metrics (Return prediction)
# #         self.train_mse = MeanSquaredError()
# #         self.val_mse = MeanSquaredError()
# #         self.test_mse = MeanSquaredError()
        
# #         self.train_mae = MeanAbsoluteError()
# #         self.val_mae = MeanAbsoluteError()
# #         self.test_mae = MeanAbsoluteError()
        
# #         self.train_r2 = R2Score()
# #         self.val_r2 = R2Score()
# #         self.test_r2 = R2Score()
    
# #     def forward(self, x, edge_index):
# #         """Forward pass through FinGAT model"""
# #         return self.model(x, edge_index)
    
# #     def _shared_step(self, batch, batch_idx: int, stage: str):
# #         """Shared step for train/val/test"""
        
# #         # Unpack batch
# #         x, edge_index, y = batch.x, batch.edge_index, batch.y
        
# #         # Forward pass
# #         returns_pred, movement_pred, ranking_scores, embeddings = self.forward(x, edge_index)
        
# #         # Extract targets
# #         returns_target = y[:, 0]  # Regression target
# #         movement_target = y[:, 1].long()  # Classification target (0/1)
        
# #         # Calculate losses
# #         regression_loss = F.mse_loss(returns_pred.squeeze(), returns_target)
# #         classification_loss = F.cross_entropy(movement_pred, movement_target)
# #         ranking_loss = self._pairwise_ranking_loss(ranking_scores.squeeze(), returns_target)
        
# #         # Combined loss (you can adjust weights)
# #         total_loss = 0.4 * regression_loss + 0.3 * classification_loss + 0.3 * ranking_loss
        
# #         # Calculate metrics
# #         movement_predictions = torch.argmax(movement_pred, dim=1)
        
# #         # Log metrics based on stage
# #         if stage == 'train':
# #             # Training metrics
# #             self.train_accuracy(movement_predictions, movement_target)
# #             self.train_mse(returns_pred.squeeze(), returns_target)
# #             self.train_mae(returns_pred.squeeze(), returns_target)
# #             self.train_r2(returns_pred.squeeze(), returns_target)
            
# #             # Log training metrics
# #             self.log(f'{stage}_loss', total_loss, on_epoch=True, prog_bar=True)
# #             self.log(f'{stage}_regression_loss', regression_loss, on_epoch=True)
# #             self.log(f'{stage}_classification_loss', classification_loss, on_epoch=True)
# #             self.log(f'{stage}_ranking_loss', ranking_loss, on_epoch=True)
# #             self.log(f'{stage}_accuracy', self.train_accuracy, on_epoch=True, prog_bar=True)
# #             self.log(f'{stage}_mse', self.train_mse, on_epoch=True)
# #             self.log(f'{stage}_mae', self.train_mae, on_epoch=True)
# #             self.log(f'{stage}_r2', self.train_r2, on_epoch=True)
            
# #         elif stage == 'val':
# #             # Validation metrics
# #             self.val_accuracy(movement_predictions, movement_target)
# #             self.val_mse(returns_pred.squeeze(), returns_target)
# #             self.val_mae(returns_pred.squeeze(), returns_target)
# #             self.val_r2(returns_pred.squeeze(), returns_target)
            
# #             # Log validation metrics
# #             self.log(f'{stage}_loss', total_loss, on_epoch=True, prog_bar=True)
# #             self.log(f'{stage}_regression_loss', regression_loss, on_epoch=True)
# #             self.log(f'{stage}_classification_loss', classification_loss, on_epoch=True)
# #             self.log(f'{stage}_ranking_loss', ranking_loss, on_epoch=True)
# #             self.log(f'{stage}_accuracy', self.val_accuracy, on_epoch=True, prog_bar=True)
# #             self.log(f'{stage}_mse', self.val_mse, on_epoch=True)
# #             self.log(f'{stage}_mae', self.val_mae, on_epoch=True)
# #             self.log(f'{stage}_r2', self.val_r2, on_epoch=True)
            
# #             # Store outputs for top-K analysis
# #             self.validation_outputs.append({
# #                 'returns_pred': returns_pred.detach().cpu(),
# #                 'movement_pred': movement_predictions.detach().cpu(),
# #                 'ranking_scores': ranking_scores.detach().cpu(),
# #                 'returns_target': returns_target.detach().cpu(),
# #                 'movement_target': movement_target.detach().cpu()
# #             })
            
# #         elif stage == 'test':
# #             # Test metrics
# #             self.test_accuracy(movement_predictions, movement_target)
# #             self.test_mse(returns_pred.squeeze(), returns_target)
# #             self.test_mae(returns_pred.squeeze(), returns_target)
# #             self.test_r2(returns_pred.squeeze(), returns_target)
            
# #             # Log test metrics
# #             self.log(f'{stage}_loss', total_loss, on_epoch=True)
# #             self.log(f'{stage}_regression_loss', regression_loss, on_epoch=True)
# #             self.log(f'{stage}_classification_loss', classification_loss, on_epoch=True)
# #             self.log(f'{stage}_ranking_loss', ranking_loss, on_epoch=True)
# #             self.log(f'{stage}_accuracy', self.test_accuracy, on_epoch=True)
# #             self.log(f'{stage}_mse', self.test_mse, on_epoch=True)
# #             self.log(f'{stage}_mae', self.test_mae, on_epoch=True)
# #             self.log(f'{stage}_r2', self.test_r2, on_epoch=True)
            
# #             # Store outputs for final top-K analysis
# #             self.test_outputs.append({
# #                 'returns_pred': returns_pred.detach().cpu(),
# #                 'movement_pred': movement_predictions.detach().cpu(),
# #                 'ranking_scores': ranking_scores.detach().cpu(),
# #                 'returns_target': returns_target.detach().cpu(),
# #                 'movement_target': movement_target.detach().cpu()
# #             })
        
# #         return total_loss
    
# #     def training_step(self, batch, batch_idx):
# #         """Training step"""
# #         return self._shared_step(batch, batch_idx, 'train')
    
# #     def validation_step(self, batch, batch_idx):
# #         """Validation step"""
# #         return self._shared_step(batch, batch_idx, 'val')
    
# #     def test_step(self, batch, batch_idx):
# #         """Test step"""
# #         return self._shared_step(batch, batch_idx, 'test')
    
# #     def on_validation_epoch_end(self):
# #         """Calculate top-K metrics at end of validation epoch"""
# #         if self.validation_outputs:
# #             self._calculate_and_log_topk_metrics(self.validation_outputs, 'val')
# #             self.validation_outputs.clear()
    
# #     def on_test_epoch_end(self):
# #         """Calculate top-K metrics at end of test epoch"""
# #         if self.test_outputs:
# #             self._calculate_and_log_topk_metrics(self.test_outputs, 'test')
            
# #             # Print top-K stock recommendations
# #             self._print_top_k_recommendations()
    
# #     def _calculate_and_log_topk_metrics(self, outputs: List[Dict], stage: str):
# #         """Calculate and log top-K metrics"""
        
# #         # Combine all outputs
# #         all_ranking_scores = torch.cat([out['ranking_scores'].squeeze() for out in outputs])
# #         all_returns_target = torch.cat([out['returns_target'] for out in outputs])
# #         all_movement_target = torch.cat([out['movement_target'] for out in outputs])
        
# #         # Calculate Top-K metrics
# #         for k in [5, 10, 20]:
# #             k_actual = min(k, len(all_ranking_scores))
# #             if k_actual > 0:
# #                 # Get top-K indices
# #                 top_k_indices = torch.topk(all_ranking_scores, k=k_actual).indices
                
# #                 # Check how many of top-K stocks actually went up
# #                 top_k_movements = all_movement_target[top_k_indices]
# #                 precision_at_k = (top_k_movements == 1).float().mean()
                
# #                 # Check average return of top-K stocks
# #                 top_k_returns = all_returns_target[top_k_indices]
# #                 avg_return_top_k = top_k_returns.mean()
                
# #                 # Log metrics
# #                 self.log(f'{stage}_precision@{k}', precision_at_k, on_epoch=True)
# #                 self.log(f'{stage}_avg_return@{k}', avg_return_top_k, on_epoch=True)
        
# #         # Calculate MRR (Mean Reciprocal Rank)
# #         mrr = self._calculate_mrr(all_ranking_scores, all_returns_target)
# #         self.log(f'{stage}_mrr', mrr, on_epoch=True, prog_bar=True)
    
# #     def _print_top_k_recommendations(self):
# #         """Print top-K stock recommendations with details"""
        
# #         if not self.test_outputs:
# #             return
            
# #         # Get final predictions
# #         all_ranking_scores = torch.cat([out['ranking_scores'].squeeze() for out in self.test_outputs])
# #         all_returns_pred = torch.cat([out['returns_pred'].squeeze() for out in self.test_outputs])
# #         all_movement_pred = torch.cat([out['movement_pred'] for out in self.test_outputs])
# #         all_returns_target = torch.cat([out['returns_target'] for out in self.test_outputs])
# #         all_movement_target = torch.cat([out['movement_target'] for out in self.test_outputs])
        
# #         # Get stock tickers
# #         tickers = list(self.metadata['ticker_to_idx'].keys())
        
# #         print("\n" + "="*80)
# #         print("🏆 TOP-K STOCK RECOMMENDATIONS")
# #         print("="*80)
        
# #         for k in [5, 10]:
# #             k_actual = min(k, len(all_ranking_scores))
# #             if k_actual > 0:
# #                 print(f"\n📊 **TOP-{k} RECOMMENDED STOCKS:**")
# #                 print("-"*70)
                
# #                 # Get top-K indices
# #                 top_k_indices = torch.topk(all_ranking_scores, k=k_actual).indices
                
# #                 print(f"{'Rank':<4} {'Ticker':<8} {'Pred Dir':<8} {'Act Dir':<8} {'Pred Ret':<10} {'Act Ret':<10} {'Score':<8} {'✓'}")
# #                 print("-"*70)
                
# #                 for i, idx in enumerate(top_k_indices):
# #                     ticker = tickers[idx] if idx < len(tickers) else f"Stock_{idx}"
                    
# #                     # Predictions
# #                     pred_direction = "UP" if all_movement_pred[idx] == 1 else "DOWN"
# #                     pred_return = f"{all_returns_pred[idx].item():.3f}"
# #                     ranking_score = f"{all_ranking_scores[idx].item():.3f}"
                    
# #                     # Actual
# #                     actual_direction = "UP" if all_movement_target[idx] == 1 else "DOWN"
# #                     actual_return = f"{all_returns_target[idx].item():.3f}"
                    
# #                     # Correctness indicators
# #                     dir_correct = "✅" if pred_direction == actual_direction else "❌"
                    
# #                     print(f"{i+1:<4} {ticker:<8} {pred_direction:<8} {actual_direction:<8} {pred_return:<10} {actual_return:<10} {ranking_score:<8} {dir_correct}")
    
# #     def _pairwise_ranking_loss(self, scores, targets):
# #         """Calculate pairwise ranking loss"""
# #         if len(scores) < 2:
# #             return torch.tensor(0.0, device=scores.device)
            
# #         # Create pairwise differences
# #         score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)  # [N, N]
# #         target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)  # [N, N]
        
# #         # Ranking loss: encourage higher scores for higher returns
# #         loss = torch.clamp(0.1 - score_diff * torch.sign(target_diff), min=0)
        
# #         # Only consider pairs where targets are different
# #         mask = (target_diff != 0).float()
# #         if mask.sum() > 0:
# #             loss = (loss * mask).sum() / mask.sum()
# #         else:
# #             loss = torch.tensor(0.0, device=scores.device)
        
# #         return loss
    
# #     def _calculate_mrr(self, scores, targets):
# #         """Calculate Mean Reciprocal Rank"""
# #         if len(scores) == 0:
# #             return torch.tensor(0.0)
            
# #         # Sort by predicted scores (descending)
# #         sorted_indices = torch.argsort(scores, descending=True)
        
# #         # Find the best actual stock (highest return)
# #         best_stock_idx = torch.argmax(targets)
        
# #         # Find rank of best stock in our predictions
# #         rank_of_best = (sorted_indices == best_stock_idx).nonzero(as_tuple=True)[0]
        
# #         if len(rank_of_best) > 0:
# #             return 1.0 / (rank_of_best[0].float() + 1)
# #         else:
# #             return torch.tensor(0.0)
    
# #     def configure_optimizers(self):
# #         """Configure optimizer and scheduler"""
# #         optimizer = torch.optim.AdamW(
# #             self.parameters(),
# #             lr=self.config['training']['learning_rate'],
# #             weight_decay=self.config['training']['weight_decay']
# #         )
        
# #         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
# #             optimizer, 
# #             mode='max',
# #             factor=0.5, 
# #             patience=10
# #         )
        
# #         return {
# #             'optimizer': optimizer,
# #             'lr_scheduler': {
# #                 'scheduler': scheduler,
# #                 'monitor': 'val_mrr'
# #             }
# #         }




# """
# FIXED FinGAT Lightning Module with Learnable Loss Balancing
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pytorch_lightning as pl
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
# from torch_geometric.nn import GATv2Conv, global_mean_pool
# from torchmetrics import Accuracy, MeanSquaredError, MeanAbsoluteError, R2Score
# from typing import Dict, Any, Tuple, List, Optional
# import numpy as np


# class ImprovedFinGAT(nn.Module):
#     """Fixed FinGAT with proper output scaling and loss balancing"""
    
#     def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4, 
#                  num_layers: int = 2, dropout: float = 0.3):
#         super().__init__()
        
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         self.num_layers = num_layers
#         self.dropout = dropout
        
#         # ✅ FIX 1: Learnable multi-task loss weights
#         self.log_vars = nn.Parameter(torch.zeros(3))  # [regression, classification, ranking]
        
#         # Input normalization
#         self.feature_norm = nn.LayerNorm(input_dim)
#         self.input_projection = nn.Linear(input_dim, hidden_dim)
        
#         # GAT layers with proper dimensions
#         self.gat_layers = nn.ModuleList()
#         current_dim = hidden_dim
        
#         for i in range(num_layers):
#             if i == 0:
#                 in_dim = hidden_dim
#             else:
#                 in_dim = hidden_dim * num_heads
                
#             concat = (i < num_layers - 1)  # Last layer doesn't concat
#             out_heads = num_heads if concat else 1
            
#             self.gat_layers.append(
#                 GATv2Conv(in_dim, hidden_dim, heads=out_heads, 
#                          dropout=dropout, concat=concat, add_self_loops=True)
#             )
        
#         # Layer normalization for each GAT layer
#         self.layer_norms = nn.ModuleList([
#             nn.LayerNorm(hidden_dim * num_heads if i < num_layers - 1 else hidden_dim)
#             for i in range(num_layers)
#         ])
        
#         # ✅ FIX 2: Task-specific feature extractors
#         self.regression_feature = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         self.classification_feature = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         self.ranking_feature = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         # ✅ FIX 3: Prediction heads with proper output scaling
#         self.regression_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout * 0.4),
#             nn.Linear(hidden_dim // 2, hidden_dim // 4),
#             nn.ReLU(),
#             nn.Dropout(dropout * 0.2),
#             nn.Linear(hidden_dim // 4, 1),
#             nn.Tanh()  # ✅ Bound to [-1, 1]
#         )
        
#         self.classification_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout * 0.4),
#             nn.Linear(hidden_dim // 2, 2)
#         )
        
#         self.ranking_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout * 0.4),
#             nn.Linear(hidden_dim // 2, 1),
#             nn.Sigmoid()  # ✅ Normalize to [0, 1]
#         )
        
#         # Initialize weights properly
#         self._init_weights()
        
#         # ✅ FIX 4: Initialize log_vars to balance losses
#         with torch.no_grad():
#             self.log_vars[0] = 0.0   # regression
#             self.log_vars[1] = -2.0  # classification (reduce influence)
#             self.log_vars[2] = -1.0  # ranking
    
#     def _init_weights(self):
#         """Proper weight initialization"""
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 if hasattr(module, 'out_features') and module.out_features == 1:
#                     nn.init.xavier_uniform_(module.weight)
#                 else:
#                     nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#             elif isinstance(module, nn.LayerNorm):
#                 nn.init.constant_(module.bias, 0)
#                 nn.init.constant_(module.weight, 1.0)
    
#     def forward(self, x, edge_index):
#         """Forward pass with proper scaling"""
        
#         # Feature normalization
#         x = self.feature_norm(x)
#         x = self.input_projection(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout * 0.5, training=self.training)
        
#         # GAT layers with residual connections
#         for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
#             x_residual = x
#             x = gat_layer(x, edge_index)
#             x = layer_norm(x)
#             x = F.elu(x)
            
#             if i < len(self.gat_layers) - 1:
#                 x = F.dropout(x, p=self.dropout, training=self.training)
        
#         embeddings = x
        
#         # Task-specific feature extraction
#         reg_features = self.regression_feature(embeddings)
#         clf_features = self.classification_feature(embeddings)
#         rank_features = self.ranking_feature(embeddings)
        
#         # ✅ FIX 5: Scale regression predictions to realistic range
#         returns_pred = self.regression_head(reg_features) * 0.1  # Scale to ±10%
#         movement_pred = self.classification_head(clf_features)
#         ranking_scores = self.ranking_head(rank_features)
        
#         return returns_pred, movement_pred, ranking_scores, embeddings
    
#     def compute_balanced_loss(self, reg_loss, clf_loss, rank_loss):
#         """
#         ✅ FIX 6: Uncertainty-based multi-task loss balancing
#         """
#         precision_reg = torch.exp(-self.log_vars[0])
#         precision_clf = torch.exp(-self.log_vars[1])
#         precision_rank = torch.exp(-self.log_vars[2])
        
#         weighted_reg = precision_reg * reg_loss + self.log_vars[0]
#         weighted_clf = precision_clf * clf_loss + self.log_vars[1]
#         weighted_rank = precision_rank * rank_loss + self.log_vars[2]
        
#         total_loss = weighted_reg + weighted_clf + weighted_rank
        
#         return total_loss, {
#             'weighted_reg': weighted_reg.item(),
#             'weighted_clf': weighted_clf.item(),
#             'weighted_rank': weighted_rank.item(),
#             'precision_reg': precision_reg.item(),
#             'precision_clf': precision_clf.item(),
#             'precision_rank': precision_rank.item()
#         }


# class FinGATDataModule(pl.LightningDataModule):
#     """PyTorch Lightning Data Module for FinGAT"""
    
#     def __init__(self, config: Dict, data: Data, metadata: Dict, batch_size: int = 1):
#         super().__init__()
#         self.config = config
#         self.data = data
#         self.metadata = metadata
#         self.batch_size = batch_size
        
#     def setup(self, stage: Optional[str] = None):
#         """Split data into train/val/test"""
#         if stage == "fit" or stage is None:
#             self.train_data = self.data
#             self.val_data = self.data
            
#         if stage == "test" or stage is None:
#             self.test_data = self.data
    
#     def train_dataloader(self):
#         return DataLoader([self.train_data], batch_size=self.batch_size, shuffle=False)
    
#     def val_dataloader(self):
#         return DataLoader([self.val_data], batch_size=self.batch_size, shuffle=False)
    
#     def test_dataloader(self):
#         return DataLoader([self.test_data], batch_size=self.batch_size, shuffle=False)


# class FinGATLightningModule(pl.LightningModule):
#     """FIXED Lightning module with learnable loss balancing"""
    
#     def __init__(self, config: Dict, metadata: Dict):
#         super().__init__()
#         self.save_hyperparameters()
        
#         self.config = config
#         self.metadata = metadata
        
#         # ✅ Use improved model
#         self.model = ImprovedFinGAT(
#             input_dim=metadata['num_features'],
#             hidden_dim=config['model']['hidden_dim'],
#             num_heads=config['model']['num_heads'],
#             num_layers=config['model'].get('num_layers', 2),
#             dropout=config['model'].get('dropout', 0.3)
#         )
        
#         self._init_metrics()
#         self.validation_outputs = []
#         self.test_outputs = []
        
#     def _init_metrics(self):
#         """Initialize all tracking metrics"""
#         self.train_accuracy = Accuracy(task='binary')
#         self.val_accuracy = Accuracy(task='binary')
#         self.test_accuracy = Accuracy(task='binary')
        
#         self.train_mse = MeanSquaredError()
#         self.val_mse = MeanSquaredError()
#         self.test_mse = MeanSquaredError()
        
#         self.train_mae = MeanAbsoluteError()
#         self.val_mae = MeanAbsoluteError()
#         self.test_mae = MeanAbsoluteError()
        
#         self.train_r2 = R2Score()
#         self.val_r2 = R2Score()
#         self.test_r2 = R2Score()
    
#     def forward(self, x, edge_index):
#         return self.model(x, edge_index)
    
#     def _shared_step(self, batch, batch_idx: int, stage: str):
#         """✅ FIXED shared step with balanced loss"""
        
#         x, edge_index, y = batch.x, batch.edge_index, batch.y
        
#         # Forward pass
#         returns_pred, movement_pred, ranking_scores, embeddings = self.forward(x, edge_index)
        
#         # Extract targets
#         returns_target = y[:, 0]
#         movement_target = y[:, 1].long()
        
#         # Calculate individual losses
#         regression_loss = F.mse_loss(returns_pred.squeeze(), returns_target)
#         classification_loss = F.cross_entropy(movement_pred, movement_target)
#         ranking_loss = self._pairwise_ranking_loss(ranking_scores.squeeze(), returns_target)
        
#         # ✅ FIX 7: Use learnable balanced loss
#         total_loss, loss_dict = self.model.compute_balanced_loss(
#             regression_loss, classification_loss, ranking_loss
#         )
        
#         # Calculate metrics
#         movement_predictions = torch.argmax(movement_pred, dim=1)
        
#         # Log metrics based on stage
#         if stage == 'train':
#             self.train_accuracy(movement_predictions, movement_target)
#             self.train_mse(returns_pred.squeeze(), returns_target)
#             self.train_mae(returns_pred.squeeze(), returns_target)
#             self.train_r2(returns_pred.squeeze(), returns_target)
            
#             self.log(f'{stage}_loss', total_loss, on_epoch=True, prog_bar=True)
#             self.log(f'{stage}_regression_loss', regression_loss, on_epoch=True)
#             self.log(f'{stage}_classification_loss', classification_loss, on_epoch=True)
#             self.log(f'{stage}_ranking_loss', ranking_loss, on_epoch=True)
#             self.log(f'{stage}_accuracy', self.train_accuracy, on_epoch=True, prog_bar=True)
#             self.log(f'{stage}_mse', self.train_mse, on_epoch=True)
#             self.log(f'{stage}_mae', self.train_mae, on_epoch=True)
#             self.log(f'{stage}_r2', self.train_r2, on_epoch=True)
            
#             # ✅ Log learned loss weights
#             self.log('reg_weight', loss_dict['precision_reg'], on_epoch=True)
#             self.log('clf_weight', loss_dict['precision_clf'], on_epoch=True)
#             self.log('rank_weight', loss_dict['precision_rank'], on_epoch=True)
            
#             # ✅ Monitor mode collapse
#             self.log('returns_pred_std', returns_pred.std(), on_epoch=True)
#             self.log('ranking_scores_std', ranking_scores.std(), on_epoch=True)
            
#         elif stage == 'val':
#             self.val_accuracy(movement_predictions, movement_target)
#             self.val_mse(returns_pred.squeeze(), returns_target)
#             self.val_mae(returns_pred.squeeze(), returns_target)
#             self.val_r2(returns_pred.squeeze(), returns_target)
            
#             self.log(f'{stage}_loss', total_loss, on_epoch=True, prog_bar=True)
#             self.log(f'{stage}_regression_loss', regression_loss, on_epoch=True)
#             self.log(f'{stage}_classification_loss', classification_loss, on_epoch=True)
#             self.log(f'{stage}_ranking_loss', ranking_loss, on_epoch=True)
#             self.log(f'{stage}_accuracy', self.val_accuracy, on_epoch=True, prog_bar=True)
#             self.log(f'{stage}_mse', self.val_mse, on_epoch=True)
#             self.log(f'{stage}_mae', self.val_mae, on_epoch=True)
#             self.log(f'{stage}_r2', self.val_r2, on_epoch=True)
            
#             self.validation_outputs.append({
#                 'returns_pred': returns_pred.detach().cpu(),
#                 'movement_pred': movement_predictions.detach().cpu(),
#                 'ranking_scores': ranking_scores.detach().cpu(),
#                 'returns_target': returns_target.detach().cpu(),
#                 'movement_target': movement_target.detach().cpu()
#             })
            
#         elif stage == 'test':
#             self.test_accuracy(movement_predictions, movement_target)
#             self.test_mse(returns_pred.squeeze(), returns_target)
#             self.test_mae(returns_pred.squeeze(), returns_target)
#             self.test_r2(returns_pred.squeeze(), returns_target)
            
#             self.log(f'{stage}_loss', total_loss, on_epoch=True)
#             self.log(f'{stage}_regression_loss', regression_loss, on_epoch=True)
#             self.log(f'{stage}_classification_loss', classification_loss, on_epoch=True)
#             self.log(f'{stage}_ranking_loss', ranking_loss, on_epoch=True)
#             self.log(f'{stage}_accuracy', self.test_accuracy, on_epoch=True)
#             self.log(f'{stage}_mse', self.test_mse, on_epoch=True)
#             self.log(f'{stage}_mae', self.test_mae, on_epoch=True)
#             self.log(f'{stage}_r2', self.test_r2, on_epoch=True)
            
#             self.test_outputs.append({
#                 'returns_pred': returns_pred.detach().cpu(),
#                 'movement_pred': movement_predictions.detach().cpu(),
#                 'ranking_scores': ranking_scores.detach().cpu(),
#                 'returns_target': returns_target.detach().cpu(),
#                 'movement_target': movement_target.detach().cpu()
#             })
        
#         return total_loss
    
#     def training_step(self, batch, batch_idx):
#         return self._shared_step(batch, batch_idx, 'train')
    
#     def validation_step(self, batch, batch_idx):
#         return self._shared_step(batch, batch_idx, 'val')
    
#     def test_step(self, batch, batch_idx):
#         return self._shared_step(batch, batch_idx, 'test')
    
#     def on_validation_epoch_end(self):
#         if self.validation_outputs:
#             self._calculate_and_log_topk_metrics(self.validation_outputs, 'val')
#             self.validation_outputs.clear()
    
#     def on_test_epoch_end(self):
#         if self.test_outputs:
#             self._calculate_and_log_topk_metrics(self.test_outputs, 'test')
#             self._print_top_k_recommendations()
    
#     def _calculate_and_log_topk_metrics(self, outputs: List[Dict], stage: str):
#         all_ranking_scores = torch.cat([out['ranking_scores'].squeeze() for out in outputs])
#         all_returns_target = torch.cat([out['returns_target'] for out in outputs])
#         all_movement_target = torch.cat([out['movement_target'] for out in outputs])
        
#         for k in [5, 10, 20]:
#             k_actual = min(k, len(all_ranking_scores))
#             if k_actual > 0:
#                 top_k_indices = torch.topk(all_ranking_scores, k=k_actual).indices
#                 top_k_movements = all_movement_target[top_k_indices]
#                 precision_at_k = (top_k_movements == 1).float().mean()
#                 top_k_returns = all_returns_target[top_k_indices]
#                 avg_return_top_k = top_k_returns.mean()
                
#                 self.log(f'{stage}_precision@{k}', precision_at_k, on_epoch=True)
#                 self.log(f'{stage}_avg_return@{k}', avg_return_top_k, on_epoch=True)
        
#         mrr = self._calculate_mrr(all_ranking_scores, all_returns_target)
#         self.log(f'{stage}_mrr', mrr, on_epoch=True, prog_bar=True)
    
#     def _print_top_k_recommendations(self):
#         if not self.test_outputs:
#             return
            
#         all_ranking_scores = torch.cat([out['ranking_scores'].squeeze() for out in self.test_outputs])
#         all_returns_pred = torch.cat([out['returns_pred'].squeeze() for out in self.test_outputs])
#         all_movement_pred = torch.cat([out['movement_pred'] for out in self.test_outputs])
#         all_returns_target = torch.cat([out['returns_target'] for out in self.test_outputs])
#         all_movement_target = torch.cat([out['movement_target'] for out in self.test_outputs])
        
#         tickers = list(self.metadata['ticker_to_idx'].keys())
        
#         print("\n" + "="*80)
#         print("🏆 TOP-K STOCK RECOMMENDATIONS")
#         print("="*80)
        
#         for k in [5, 10]:
#             k_actual = min(k, len(all_ranking_scores))
#             if k_actual > 0:
#                 print(f"\n📊 **TOP-{k} RECOMMENDED STOCKS:**")
#                 print("-"*70)
                
#                 top_k_indices = torch.topk(all_ranking_scores, k=k_actual).indices
                
#                 print(f"{'Rank':<4} {'Ticker':<8} {'Pred Dir':<8} {'Act Dir':<8} {'Pred Ret':<10} {'Act Ret':<10} {'Score':<8} {'✓'}")
#                 print("-"*70)
                
#                 for i, idx in enumerate(top_k_indices):
#                     ticker = tickers[idx] if idx < len(tickers) else f"Stock_{idx}"
#                     pred_direction = "UP" if all_movement_pred[idx] == 1 else "DOWN"
#                     pred_return = f"{all_returns_pred[idx].item():.3f}"
#                     ranking_score = f"{all_ranking_scores[idx].item():.3f}"
#                     actual_direction = "UP" if all_movement_target[idx] == 1 else "DOWN"
#                     actual_return = f"{all_returns_target[idx].item():.3f}"
#                     dir_correct = "✅" if pred_direction == actual_direction else "❌"
                    
#                     print(f"{i+1:<4} {ticker:<8} {pred_direction:<8} {actual_direction:<8} {pred_return:<10} {actual_return:<10} {ranking_score:<8} {dir_correct}")
    
#     def _pairwise_ranking_loss(self, scores, targets):
#         if len(scores) < 2:
#             return torch.tensor(0.0, device=scores.device)
            
#         score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
#         target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
#         loss = torch.clamp(0.1 - score_diff * torch.sign(target_diff), min=0)
#         mask = (target_diff != 0).float()
        
#         if mask.sum() > 0:
#             loss = (loss * mask).sum() / mask.sum()
#         else:
#             loss = torch.tensor(0.0, device=scores.device)
        
#         return loss
    
#     def _calculate_mrr(self, scores, targets):
#         if len(scores) == 0:
#             return torch.tensor(0.0)
            
#         sorted_indices = torch.argsort(scores, descending=True)
#         best_stock_idx = torch.argmax(targets)
#         rank_of_best = (sorted_indices == best_stock_idx).nonzero(as_tuple=True)[0]
        
#         if len(rank_of_best) > 0:
#             return 1.0 / (rank_of_best[0].float() + 1)
#         else:
#             return torch.tensor(0.0)
    
#     def configure_optimizers(self):
#         # ✅ Higher learning rate for faster convergence
#         optimizer = torch.optim.AdamW(
#             self.parameters(),
#             lr=5e-4,  # Increased from 1e-3
#             weight_decay=1e-4
#         )
        
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, 
#             mode='max',
#             factor=0.5, 
#             patience=10
#         )
        
#         return {
#             'optimizer': optimizer,
#             'lr_scheduler': {
#                 'scheduler': scheduler,
#                 'monitor': 'val_mrr'
#             }
#         }




"""
FINAL FIXED FinGAT Lightning Module with Proper Data Splits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv
from torchmetrics import Accuracy, MeanSquaredError, MeanAbsoluteError, R2Score
from typing import Dict, Tuple, List, Optional


class ImprovedFinGAT(nn.Module):
    """Fixed FinGAT with proper output scaling and loss balancing"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Learnable multi-task loss weights
        self.log_vars = nn.Parameter(torch.zeros(3))
        
        # Input normalization
        self.feature_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            concat = (i < num_layers - 1)
            out_heads = num_heads if concat else 1
            
            self.gat_layers.append(
                GATv2Conv(in_dim, hidden_dim, heads=out_heads, 
                         dropout=dropout, concat=concat, add_self_loops=True)
            )
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * num_heads if i < num_layers - 1 else hidden_dim)
            for i in range(num_layers)
        ])
        
        # Task-specific feature extractors
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
        
        # Prediction heads with proper output scaling
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
        
        self._init_weights()
        
        # Line 91-93 in ImprovedFinGAT.__init__()
        with torch.no_grad():
            self.log_vars[0] = -1.0  # Increase regression weight
            self.log_vars[1] = -2.0  # Keep classification lower  
            self.log_vars[2] = -1.5  # Keep ranking moderate

    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'out_features') and module.out_features == 1:
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
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
    
    def compute_balanced_loss(self, reg_loss, clf_loss, rank_loss):
        precision_reg = torch.exp(-self.log_vars[0])
        precision_clf = torch.exp(-self.log_vars[1])
        precision_rank = torch.exp(-self.log_vars[2])
        
        weighted_reg = precision_reg * reg_loss + self.log_vars[0]
        weighted_clf = precision_clf * clf_loss + self.log_vars[1]
        weighted_rank = precision_rank * rank_loss + self.log_vars[2]
        
        total_loss = weighted_reg + weighted_clf + weighted_rank
        
        return total_loss, {
            'weighted_reg': weighted_reg.item(),
            'weighted_clf': weighted_clf.item(),
            'weighted_rank': weighted_rank.item(),
            'precision_reg': precision_reg.item(),
            'precision_clf': precision_clf.item(),
            'precision_rank': precision_rank.item()
        }


class FinGATDataModule(pl.LightningDataModule):
    """✅ FIXED: Data Module with proper train/val/test splits"""
    
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


class FinGATLightningModule(pl.LightningModule):
    """FIXED Lightning module with learnable loss balancing"""
    
    def __init__(self, config: Dict, metadata: Dict):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.metadata = metadata
        
        self.model = ImprovedFinGAT(
            input_dim=metadata['num_features'],
            hidden_dim=config['model']['hidden_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model'].get('num_layers', 2),
            dropout=config['model'].get('dropout', 0.3)
        )
        
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
    
    def forward(self, x, edge_index):
        return self.model(x, edge_index)
    
    def _shared_step(self, batch, batch_idx: int, stage: str):
        x, edge_index, y = batch.x, batch.edge_index, batch.y
        
        returns_pred, movement_pred, ranking_scores, embeddings = self.forward(x, edge_index)
        
        returns_target = y[:, 0]
        movement_target = y[:, 1].long()
        
        regression_loss = F.mse_loss(returns_pred.squeeze(), returns_target)
        classification_loss = F.cross_entropy(movement_pred, movement_target)
        ranking_loss = self._pairwise_ranking_loss(ranking_scores.squeeze(), returns_target)
        
        total_loss, loss_dict = self.model.compute_balanced_loss(
            regression_loss, classification_loss, ranking_loss
        )
        
        movement_predictions = torch.argmax(movement_pred, dim=1)
        
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
            
            self.log('reg_weight', loss_dict['precision_reg'], on_epoch=True)
            self.log('clf_weight', loss_dict['precision_clf'], on_epoch=True)
            self.log('rank_weight', loss_dict['precision_rank'], on_epoch=True)
            
            self.log('returns_pred_std', returns_pred.std(), on_epoch=True)
            self.log('ranking_scores_std', ranking_scores.std(), on_epoch=True)
            
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
        
        for k in [5, 10]:
            k_actual = min(k, len(all_ranking_scores))
            if k_actual > 0:
                print(f"\n📊 **TOP-{k} RECOMMENDED STOCKS:**")
                print("-"*70)
                
                top_k_indices = torch.topk(all_ranking_scores, k=k_actual).indices
                
                print(f"{'Rank':<4} {'Ticker':<8} {'Pred Dir':<8} {'Act Dir':<8} {'Pred Ret':<10} {'Act Ret':<10} {'Score':<8} {'✓'}")
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
