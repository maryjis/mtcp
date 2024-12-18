# Standard libraries
from typing import Tuple, Union

# Third-party libraries
import torch
import torch.nn as nn
from monai.networks.nets import resnet10


class MRIEncoder(nn.Module):
    def __init__(
        self, in_channels: int, embedding_dim: int = 512, projection_head: bool = True
    ) -> None:
        super().__init__()
        if projection_head:
            self.projection_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, int(embedding_dim / 2)),
            )

        self.encoder = resnet10(
            spatial_dims=3, n_input_channels=in_channels, num_classes=1
        )
        if embedding_dim != 512:
            self.encoder.fc = nn.Sequential(
                nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, embedding_dim)
            )
        else:
            self.encoder.fc = nn.Identity()

    def forward(
        self, x: torch.tensor
    ) -> Union[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        x = self.encoder(x)
        if hasattr(self, "projection_head"):
            return self.projection_head(x), x
        else:
            return x
