import torch
import torch.nn as nn
import torchvision
from typing import Tuple, Union
from einops import repeat
from .func import Transformer
from einops.layers.torch import Rearrange


class ResNetWrapperSimCLR(nn.Module):
    """
    A wrapper for the ResNet34 model with a projection head for SimCLR.
    """

    def __init__(self, out_dim: int, projection_head: bool = True) -> None:
        super().__init__()
        self.encoder = torchvision.models.resnet34(pretrained=False)
        self.encoder.fc = nn.Identity()
        if projection_head:
            self.projection_head = nn.Sequential(
                nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, out_dim)
            )

    def forward(
        self, x: torch.tensor
    ) -> Union[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        x = self.encoder(x)
        if hasattr(self, "projection_head"):
            return self.projection_head(x), x
        else:
            return x


class WSIEncoder(nn.Module):
    """
    An attention-based encoder for WSI data.
    """

    def __init__(
        self,
        embedding_dim: int,
        depth: int,
        heads: int,
        dim: int = 512,
        pool: str = "cls",
        dim_head: int = 64,
        mlp_dim: int = 128,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        n_outputs: int = 20  # Parameter for the number of output features
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # Pooling (cls or mean)
        self.pool = pool

        # Transformation to the required size
        self.to_latent = (
            nn.Identity() if embedding_dim == dim else nn.Linear(dim, embedding_dim)
        )
        self.n_outputs = n_outputs

        # Adding a linear layer for output features
        self.output_layer = nn.Linear(embedding_dim, self.n_outputs)  # New output layer

    def forward(self, x: torch.Tensor, masks=None) -> torch.Tensor:
        x = self.layer_norm(x)
        b, n, _ = x.shape

        # Adding CLS token
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # Applying dropout
        x = self.dropout(x)

        # Passing through the transformer
        x = self.transformer(x)

        # Performing pooling (selecting the CLS token or averaging)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        
        # Transforming to the required size
        x = self.to_latent(x)

        # Passing through the output layer
        x = self.output_layer(x)

        return x
    
# class WSIEmbedding:
#      def __init__(self, num_patches):
#          self.num_patches = num_patches
         
# class WSIForMultimodalEncoder(nn.Module):
#     """
#     An attention-based encoder for WSI data.
#     """

#     def __init__(
#         self,
#         embedding_dim: int,
#         depth: int,
#         heads: int,
#         dim: int = 512,
#         pool: str = "cls",
#         dim_head: int = 64,
#         mlp_dim: int = 128,
#         dropout: float = 0.0,
#         emb_dropout: float = 0.0,
#         cfg = None
#     ) -> None:
#         super().__init__()
#         self.cfg = cfg
#         self.embeddings = WSIEmbedding(10)
        
#         self.layer_norm = nn.LayerNorm(dim)

#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

#         self.pool = pool
       
#         self.to_latent = (
#             nn.Identity() if embedding_dim == dim else nn.Linear(dim, embedding_dim)
#         )

#     def forward(self, x: torch.Tensor, masks=None) -> torch.Tensor:
#         x = self.layer_norm(x)
#         b, n, _ = x.shape

#         cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
#         x = torch.cat((cls_tokens, x), dim=1)

#         x = self.dropout(x)

#         x = self.transformer(x)

#         x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        
#         x = self.to_latent(x)

#         return x
    
#     def patchify(self, x, interpolate_pos_encoding: bool = False):
#         return x
    
#     def unpatchify(self, x, original_size = None):
#         return x 


