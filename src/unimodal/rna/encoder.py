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


# class RNAEncoder(nn.Module):
#     """
#     A vanilla encoder based on 1-d convolution for RNA data.
#     """

#     def __init__(self, embedding_dim: int, dropout: float) -> None:
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv1d(1, 64, 9, 3),
#             nn.GELU(),
#             nn.BatchNorm1d(64),
#             nn.Dropout(dropout),

#             nn.Conv1d(64, 128, 9, 3),
#             nn.GELU(),
#             nn.BatchNorm1d(128),
#             nn.Dropout(dropout),

#             nn.Conv1d(128, 256, 9, 3),
#             nn.GELU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(dropout),

#             nn.Conv1d(256, 384, 9, 3),
#             nn.GELU(),
#             nn.BatchNorm1d(384),
#             nn.Dropout(dropout),

#             nn.Conv1d(384, 512, 9, 3),
#             nn.GELU(),
#             nn.BatchNorm1d(512),
#             nn.Dropout(dropout),

#             nn.Conv1d(512, embedding_dim, 9, 3),
#             nn.GELU(),
#             nn.BatchNorm1d(embedding_dim),
#             nn.Dropout(dropout),

#             nn.AdaptiveAvgPool1d(1),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.encoder(x).squeeze(-1)
#         return x
    
    
    
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
        
import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int = 18160, embedding_dim: int = 768, dropout: float = 0.0) -> None:
        super().__init__()

        # Подогнано под ~47.05M параметров
        h1 = 2208
        h2 = 1536
        h3 = 1024
        h4 = 768
        h5 = 768

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.GELU(),
            nn.LayerNorm(h1),
            nn.Dropout(dropout),

            nn.Linear(h1, h2),
            nn.GELU(),
            nn.LayerNorm(h2),
            nn.Dropout(dropout),

            nn.Linear(h2, h3),
            nn.GELU(),
            nn.LayerNorm(h3),
            nn.Dropout(dropout),

            nn.Linear(h3, h4),
            nn.GELU(),
            nn.LayerNorm(h4),
            nn.Dropout(dropout),

            nn.Linear(h4, h5),
            nn.GELU(),
            nn.LayerNorm(h5),
            nn.Dropout(dropout),

            nn.Linear(h5, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x  


class MLPSurvival(MLPEncoder):
    def __init__(self,input_dim:int, embedding_dim: int, dropout: float, dropout_final: float, n_out: int) -> None:
        super().__init__(input_dim,embedding_dim, dropout)
        #self.projection = nn.Linear(embedding_dim, n_out)
        self.projection = nn.Sequential(nn.Dropout(dropout_final),nn.Linear(embedding_dim, n_out))
        
        
    def forward(self, x: torch.Tensor, masks=None) -> torch.Tensor:
        x = super().forward(x)
        x = self.projection(x).squeeze(-2)
        return x    

def initialise_mlp_model(cfg: DictConfig):
    return MLPSurvival(cfg.input_dim, cfg.embedding_dim, cfg.dropout,cfg.dropout_final, cfg.output_dim)

def initialise_snn_model(cfg: DictConfig):
    return None