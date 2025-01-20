import torch
import torch.nn as nn
from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEModel,
    ViTMAEDecoder,
    ViTMAEForPreTraining,
    get_1d_sincos_pos_embed_from_grid
)
from einops import rearrange
import numpy as np

import os 


class WsiMAEPatchEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.hidden_size = cfg.hidden_size
        self.num_channels = cfg.num_channels
        
        # 2D convolution to extract features from patches

        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, patches):
        # Шаг 1: Преобразуем размерность патчей
        # Объединяем батчи и патчи в одну размерность (b * n, c, h, w)
        # 1. Преобразуем данные в формат (batch_size * num_patches, num_channels, patch_size, patch_size)
        b=patches.shape[0]
        patches = rearrange(patches, 'b n c h w -> (b n) c h w')
        
        # Шаг 2: Применяем свертку
        x = self.projection(patches)
        
        
        # Шаг 3: Преобразуем обратно в (batch_size, num_patches, hidden_size)
        x = rearrange(x, '(b n) c 1 1 -> b n c', b=b)
    
        return x


class WsiMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position, and patch embeddings for 2D data (e.g., images).
    """

    def __init__(self, cfg):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))  # CLS token
        self.patch_embeddings = WsiMAEPatchEmbeddings(cfg)  # A class for embedding 2D patches
        self.num_patches = cfg.num_patches  # Number of patches
        # Fixed sin-cos embedding for position information
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, cfg.hidden_size), requires_grad=False
        )
        self.patch_size = cfg.patch_size
        self.config = cfg
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.position_embeddings.shape[-1], 
            np.arange(int(self.num_patches), dtype=np.float32)
        )
        pos_embed = np.concatenate([
            np.zeros([1, self.position_embeddings.shape[-1]]), # for CLS token
            pos_embed
        ], axis=0)
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w)

        # Initialize CLS token with normal distribution
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    def random_masking(self, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)

        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def forward(self, image_values, noise=None, interpolate_pos_encoding: bool = False):
        batch_size, num_patches, num_channels, img_height, img_width = image_values.shape
        embeddings = self.patch_embeddings(image_values)  # Embedding patches into higher dimension

        # Add position embeddings without CLS token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        # Apply random masking to the sequence
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        # Append CLS token at the beginning of the sequence
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore

class WsiMAEModel(ViTMAEModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = WsiMAEEmbeddings(config)
        self.post_init()

    def patchify(self, imgs):
        return imgs



class WsiMAEDecoderPred(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.mri_size % config.patch_size == 0, "MRI size must be divisible by patch size"
        
        # Количество патчей по осям
        self.num_patches_along_axis = config.mri_size // config.patch_size

        # Декодер: 2D транспонированная свертка
        self.projector = nn.ConvTranspose2d(
            config.decoder_hidden_size, 
            config.num_channels, 
            config.patch_size, 
            stride=config.patch_size
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Добавление размерности для свертки (необходимо для использования ConvTranspose2d)
        x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)  # [B, S, E] -> [B, S, E, 1, 1]
        
        # Перестановка: объединяем batch_size и seq_len, чтобы обработать их одновременно
        x = rearrange(x, 'b s e x y -> (b s) e x y')

        # Проход через транспонированную свертку для восстановления изображения
        x = self.projector(x)  # [B*S, E, 1, 1] -> [B*S, C, 16, 16]

        # Возврат к исходной форме: (b, s, c, h, w)
        x = rearrange(x, '(b s) c h w -> b s (c h w)', b=batch_size)

        return x
    
class WsiMAEDecoder(ViTMAEDecoder):
    def __init__(self, config, num_patches):
        super().__init__(config, num_patches)
        self.decoder_pred = WsiMAEDecoderPred(config)
        self.initialize_weights(num_patches)
        
    def initialize_weights(self, num_patches):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.decoder_pos_embed.shape[-1], np.arange(int(num_patches), dtype=np.float32))
        decoder_pos_embed = np.concatenate([
            np.zeros([1, self.decoder_pos_embed.shape[-1]]), 
            decoder_pos_embed
        ], axis=0)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)


class WsiMAEForPreTraining(ViTMAEForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # Модель ViT для извлечения эмбеддингов
        self.vit = WsiMAEModel(config)
        # Декодер для восстановления изображения
        self.decoder = WsiMAEDecoder(config, num_patches=self.vit.embeddings.num_patches)
        self.post_init()

    def patchify(self, imgs):
        return imgs
    

class WsiMaeSurvivalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.to_dict().get("is_load_pretrained", False):
            self.vit = WsiMAEModel.from_pretrained(config.pretrained_model_path, config=config)
            print(f"Pretrained model loaded from {config.pretrained_model_path}")
        else:
            self.vit = WsiMAEModel(config)
        self.projection = nn.Linear(config.hidden_size, config.output_dim)
        
    def forward(self, wsi_values, masks=None):
        x = self.vit(wsi_values)
        x = self.projection(x.last_hidden_state[:, 0, :])
        return x.squeeze(-1)
