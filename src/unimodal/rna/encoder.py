import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
from omegaconf import DictConfig

class RNAEncoder(nn.Module):
    """
    A vanilla encoder based on 1-d convolution for RNA data.
    """

    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, 9, 3),
            nn.GELU(),
            nn.BatchNorm1d(8),
            nn.Dropout(dropout),
            nn.Conv1d(8, 32, 9, 3),
            nn.GELU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, 9, 3),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, 9, 3),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, 9, 3),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Conv1d(256, embedding_dim, 9, 3),
            nn.GELU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x).squeeze(-1)
        return x
    
    
    
def initialise_rna_model(cfg: DictConfig):
    return EncoderSurvival(cfg.embedding_dim, cfg.dropout, cfg.output_dim)



class EncoderSurvival(RNAEncoder):
    def __init__(self, embedding_dim: int, dropout: float, n_out: int) -> None:
        super().__init__(embedding_dim, dropout)
        self.projection = nn.Linear(embedding_dim, n_out)
        
        
    def forward(self, x: torch.Tensor, masks=None) -> torch.Tensor:
        x = super().forward(x)
        x = self.projection(x).squeeze(-1)
        return x    
        
    