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
        n_outputs: int = 20  # Параметр для количества выходных признаков
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)

        # cls токен
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Трансформер
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # Пуллинг (cls или mean)
        self.pool = pool

        # Преобразование в нужный размер
        self.to_latent = (
            nn.Identity() if embedding_dim == dim else nn.Linear(dim, embedding_dim)
        )

        # Добавляем линейный слой для выходных признаков
        self.output_layer = nn.Linear(embedding_dim, n_outputs)  # Новый слой для выхода

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        b, n, _ = x.shape

        # Добавляем CLS токен
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # Применяем dropout
        x = self.dropout(x)

        # Пропускаем через трансформер
        x = self.transformer(x)

        # Выполняем пуллинг (выбор CLS токена или усреднение)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        
        # Преобразуем в нужный размер
        x = self.to_latent(x)

        # Проходим через выходной слой
        x = self.output_layer(x)

        return x
