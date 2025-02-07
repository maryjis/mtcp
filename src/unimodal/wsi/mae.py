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
        super().__init__()
        self.image_size = cfg.image_size  # Ожидаем одно число, например 256
        self.patch_size = cfg.patch_size  # Ожидаем одно число, например 16
        self.hidden_size = cfg.hidden_size
        self.num_channels = cfg.num_channels

        # Количество подпатчей в одном 256x256 патче
        self.num_sub_patches = (self.image_size // self.patch_size) ** 2  # 16 * 16 = 256

        # Общее число патчей после разбиения всех max_patches_per_sample
        self.total_patches = cfg.max_patches_per_sample * self.num_sub_patches

        # Свёрточный слой для получения эмбеддингов
        self.projection = nn.Conv2d(
            self.num_channels, self.hidden_size,
            kernel_size=self.patch_size, stride=self.patch_size
        )

    def _split_patches(self, wsi_tensor):
        """
        Разбивает входные патчи на подпатчи.
        Ожидаемый вход: [batch_size, max_patches_per_sample, 3, 256, 256]
        """
        b, n, c, h, w = wsi_tensor.shape  # n = max_patches_per_sample
        p = self.patch_size  # (16)

        # Проверяем, что размеры делятся без остатка
        if h % p != 0 or w % p != 0:
            raise ValueError(f"Размер изображения ({h}, {w}) не делится на ({p}, {p}) без остатка.")

        # Разбиваем каждый 256×256 патч на 16×16 подпатчи
        patches = wsi_tensor.unfold(3, p, p).unfold(4, p, p)  # [b, n, c, h//p, p, w//p, p]
        patches = patches.permute(0, 1, 3, 5, 2, 4, 6).reshape(b, n * self.num_sub_patches, c, p, p)

        return patches 

    def forward(self, wsi_tensor):
        """Разбивает входные патчи и пропускает их через эмбеддинги."""
        patches = self._split_patches(wsi_tensor)  # Разбиение на подпатчи

        # Преобразуем в (batch_size * num_patches, num_channels, patch_size, patch_size)
        b, n, c, h, w = patches.shape
        patches = rearrange(patches, 'b n c h w -> (b n) c h w')

        # Применяем свёрточный слой
        x = self.projection(patches)  # (b*n, hidden_size, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # Убираем последние размерности -> (b*n, hidden_size)

        # Преобразуем обратно в (batch_size, num_patches, hidden_size)
        x = rearrange(x, '(b n) c -> b n c', b=b, n=self.total_patches)

        return x  # (batch_size, total_patches, hidden_size)


class WsiMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position, and patch embeddings for 2D data (e.g., images).
    """

    def __init__(self, cfg):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))  # CLS token
        self.patch_embeddings = WsiMAEPatchEmbeddings(cfg)  # A class for embedding 2D patches
        self.num_patches = self.patch_embeddings.total_patches # Number of patches
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
        #print(image_values.shape)
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

    def patchify(self, imgs, interpolate_pos_encoding: bool = False):
        """
        Разбивает входные изображения на патчи.
        Ожидаемый вход: [batch_size, max_patches_per_sample, 3, image_size, image_size].
        Возвращает: [batch_size, total_patches, num_channels * patch_size * patch_size].
        """
        p = self.config.patch_size  # Размер подпатча, например, 16
        b, n, c, h, w = imgs.shape  # [batch_size, max_patches_per_sample, 3, 256, 256]

        assert h == w == self.config.image_size, "Размер изображения не совпадает с config.image_size"
        assert h % p == 0 and w % p == 0, "Размер изображения должен делиться на patch_size"

        # Разбиение изображения на патчи с помощью rearrange
        patches = rearrange(imgs, 'b n c (h p1) (w p2) -> b (n h w) c p1 p2', p1=p, p2=p)

        # Объединяем каналы и размерности патча в один вектор признаков
        patches = patches.flatten(2)  # [batch_size, total_patches, num_channels * patch_size^2]

        return patches

class WsiMAEDecoderPred(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.image_size % config.patch_size == 0, \
            f"image_size ({config.image_size}) должно делиться на patch_size ({config.patch_size}) без остатка."

        # Количество патчей по высоте и ширине
        self.num_patches_per_dim = config.image_size // config.patch_size

        # Декодер: 2D транспонированная свертка
        self.projector = nn.ConvTranspose2d(
            config.decoder_hidden_size,
            config.num_channels,
            config.patch_size,
            stride=config.patch_size
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Добавление размерности для ConvTranspose2d
        x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)  # [B, S, E] -> [B, S, E, 1, 1]
        
        # Перестановка: объединяем batch_size и seq_len, чтобы обработать их одновременно
        x = rearrange(x, 'b s e x y -> (b s) e x y')

        # Проход через транспонированную свертку для восстановления изображения
        x = self.projector(x)  # [B*S, E, 1, 1] -> [B*S, C, patch_size, patch_size]

        # Возврат к исходной форме: (B, num_patches, C, patch_size, patch_size)
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

    def patchify(self, imgs, interpolate_pos_encoding: bool = False):
        """
        Разбивает входные изображения на патчи.
        Ожидаемый вход: [batch_size, max_patches_per_sample, 3, image_size, image_size].
        Возвращает: [batch_size, total_patches, num_channels * patch_size * patch_size].
        """
        p = self.config.patch_size  # Размер подпатча, например, 16
        b, n, c, h, w = imgs.shape  # [batch_size, max_patches_per_sample, 3, 256, 256]

        assert h == w == self.config.image_size, "Размер изображения не совпадает с config.image_size"
        assert h % p == 0 and w % p == 0, "Размер изображения должен делиться на patch_size"

        # Разбиение изображения на патчи с помощью rearrange
        patches = rearrange(imgs, 'b n c (h p1) (w p2) -> b (n h w) c p1 p2', p1=p, p2=p)

        # Объединяем каналы и размерности патча в один вектор признаков
        patches = patches.flatten(2)  # [batch_size, total_patches, num_channels * patch_size^2]

        return patches

    

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
