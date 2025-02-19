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
        self.image_size = cfg.image_size
        self.patch_size = cfg.patch_size
        self.hidden_size = cfg.hidden_size
        self.num_channels = cfg.num_channels
        max_patches_per_sample = cfg.max_patches_per_sample
        
        # Количество подпатчей в одном изображении размером image_size x image_size
        self.num_sub_patches = (self.image_size // self.patch_size) ** 2
        
        # Общее число патчей после разбиения всех max_patches_per_sample
        self.total_patches = max_patches_per_sample * self.num_sub_patches

        ############################################
        # Вариант с несколькими мелкими свёртками
        # Допустим, хотим уменьшить разрешение с 256 -> 16x16, но постепенно
        # Можно сделать последовательность Conv3×3 + BatchNorm + ReLU + Pooling
        ############################################
        layers = []
        in_ch = self.num_channels
        out_ch = 64  # К примеру, начнём с 64 каналов
        # Первый блок: Conv3x3 (уменьшим размер в 2 раза пуллингом)
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # деление разрешения на 2

        # Второй блок: Conv3x3 (снова уменьшим размер в 2 раза)
        layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Третий блок (опционально, если хотим ещё меньше):
        # Можно ещё уменьшить в 2 раза размер, если нужно.
        # layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1))
        # layers.append(nn.BatchNorm2d(out_ch))
        # layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Далее "доворачиваем" до нужного hidden_size
        # Можно одним линейным слоем после global pooling или Conv, чтобы достичь нужного hidden_size
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(out_ch, self.hidden_size))

        self.projection = nn.Sequential(*layers)

    def _split_patches(self, wsi_tensor):
        """
        Разбивает входные патчи на подпатчи.
        Ожидаемый вход: [batch_size, max_patches_per_sample, 3, 256, 256]
        """
        b, n, c, h, w = wsi_tensor.shape
        p = self.patch_size

        if h % p != 0 or w % p != 0:
            raise ValueError(f"Размер изображения ({h}, {w}) не делится на ({p}, {p}) без остатка.")

        # unfold + permute + view
        patches = wsi_tensor.unfold(3, p, p).unfold(4, p, p)  # -> [b, n, c, h//p, p, w//p, p]
        patches = patches.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        patches = patches.view(b, n * self.num_sub_patches, c, p, p)

        return patches

    def forward(self, wsi_tensor):
        # Разбиваем на подпатчи
        patches = self._split_patches(wsi_tensor)  
        b, n, c, h, w = patches.shape

        # Преобразуем в (batch_size * num_patches, c, h, w)
        patches = rearrange(patches, 'b n c h w -> (b n) c h w')

        # Пропускаем через несколько мелких свёрточных слоёв + pooling
        x = self.projection(patches)  # [b*n, hidden_size]

        # Возвращаемся к (batch_size, num_patches, hidden_size)
        x = rearrange(x, '(b n) c -> b n c', b=b, n=self.total_patches)

        return x


class WsiMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position, and patch embeddings for 2D data (e.g., images).
    """
    def __init__(self, cfg):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.patch_embeddings = WsiMAEPatchEmbeddings(cfg)
        self.num_patches = self.patch_embeddings.total_patches
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, cfg.hidden_size), requires_grad=False
        )
        self.patch_size = cfg.patch_size
        self.config = cfg
        self.initialize_weights()

    def initialize_weights(self):
        # Sin-cos эмбеддинги
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.position_embeddings.shape[-1], 
            np.arange(int(self.num_patches), dtype=np.float32)
        )
        pos_embed = np.concatenate([
            np.zeros([1, self.position_embeddings.shape[-1]]),  # для CLS
            pos_embed
        ], axis=0)
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Инициализация (Xavier, normal, и т.д.)
        for m in self.patch_embeddings.projection:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    def random_masking(self, sequence, noise=None):
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, 1, ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return sequence_unmasked, mask, ids_restore

    def forward(self, image_values, noise=None, interpolate_pos_encoding: bool = False):
        batch_size, num_patches, num_channels, img_height, img_width = image_values.shape
        embeddings = self.patch_embeddings(image_values)
        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        # Мэскируем
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)
        
        # CLS token
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
        p = self.config.patch_size
        b, n, c, h, w = imgs.shape
        assert h == w == self.config.image_size, "Размер изображения не совпадает с config.image_size"
        assert h % p == 0 and w % p == 0, "Размер изображения должен делиться на patch_size"
        patches = rearrange(imgs, 'b n c (h p1) (w p2) -> b (n h w) c p1 p2', p1=p, p2=p)
        patches = patches.flatten(2)
        return patches


class WsiMAEDecoderPred(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.image_size % config.patch_size == 0
        self.num_patches_per_dim = config.image_size // config.patch_size
        self.projector = nn.ConvTranspose2d(
            config.decoder_hidden_size,
            config.num_channels,
            config.patch_size,
            stride=config.patch_size
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(-1).unsqueeze(-1) 
        x = rearrange(x, 'b s e x y -> (b s) e x y')
        x = self.projector(x)
        x = rearrange(x, '(b s) c h w -> b s (c h w)', b=batch_size)
        return x


class WsiMAEDecoder(ViTMAEDecoder):
    def __init__(self, config, num_patches):
        super().__init__(config, num_patches)
        self.decoder_pred = WsiMAEDecoderPred(config)
        self.initialize_weights(num_patches)
        
    def initialize_weights(self, num_patches):
        decoder_pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.decoder_pos_embed.shape[-1], np.arange(int(num_patches), dtype=np.float32))
        decoder_pos_embed = np.concatenate([
            np.zeros([1, self.decoder_pos_embed.shape[-1]]),
            decoder_pos_embed
        ], axis=0)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)


class WsiMAEForPreTraining(ViTMAEForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vit = WsiMAEModel(config)
        self.decoder = WsiMAEDecoder(config, num_patches=self.vit.embeddings.num_patches)
        self.post_init()

    def patchify(self, imgs, interpolate_pos_encoding: bool = False):
        p = self.config.patch_size
        b, n, c, h, w = imgs.shape
        assert h == w == self.config.image_size, "Размер изображения не совпадает с config.image_size"
        assert h % p == 0 and w % p == 0, "Размер изображения должен делиться на patch_size"
        patches = rearrange(imgs, 'b n c (h p1) (w p2) -> b (n h w) c p1 p2', p1=p, p2=p)
        patches = patches.flatten(2)
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



class WSIEmbeddings(nn.Module):
    """
    Construct the CLS token, position, and patch embeddings for 2D data (e.g., images).
    """

    def __init__(self, cfg):
        super().__init__()

        self.config = cfg
        
        self.projector = nn.Linear(cfg.input_embedding_dim, cfg.hidden_size)
    
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))  # CLS token
        
        self.num_patches = cfg.num_patches # Number of patches
        # Fixed sin-cos embedding for position information
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, cfg.hidden_size), requires_grad=False
        )
        self.patch_size = cfg.patch_size
        
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

    def forward(self, embeddings, noise=None, interpolate_pos_encoding: bool = False):
        #print(image_values.shape)
        batch_size, num_patches, hidden_size = embeddings.shape
    
        # Add position embeddings without CLS token
        embeddings = self.projector(embeddings) + self.position_embeddings[:, 1:, :]

        # Apply random masking to the sequence
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        # Append CLS token at the beginning of the sequence
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore
    


class WSIEmbeddingMAEModel(ViTMAEModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = WSIEmbeddings(config)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.post_init()
        
    def patchify(self, values, interpolate_pos_encoding: bool = False):
        return values

    def unpatchify(self, values, original_rna_size: int=None):
        return values