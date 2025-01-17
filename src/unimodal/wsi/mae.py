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

class WsiTMAEPatchEmbeddings(nn.Module):
    """
    Этот класс преобразует WSI данные (например, слайды с размером [batch_size, channels, height, width])
    в начальные скрытые состояния (patch embeddings) с размером [batch_size, seq_length, hidden_size],
    которые можно использовать для трансформера.
    """

    def __init__(self, cfg):
        super().__init__()
        img_size, patch_size = cfg.img_size, cfg.patch_size
        num_channels, hidden_size = cfg.num_channels, cfg.hidden_size
        assert img_size % patch_size == 0, "Размер изображения должен делиться на размер патча"
        num_patches = (img_size ** 2) // (patch_size ** 2)

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, wsi_values):
        batch_size, num_channels, img_size1, img_size2 = wsi_values.shape
        if num_channels != self.num_channels:
            raise ValueError("Убедитесь, что размерность канала WSI данных совпадает с указанной в конфигурации.")

        wsi_values = rearrange(
            wsi_values,
            'b c (x p_x) (y p_y) -> (b x y) c p_x p_y', 
            p_x=self.patch_size, 
            p_y=self.patch_size
        )
        x = self.projection(wsi_values)  # B*S, C, P_x, P_y -> B*S, E
        x = rearrange(x.squeeze(), '(b s) e -> b s e', b=batch_size)
        return x


class WsiMAEEmbeddings(nn.Module):
    """
    Конструирует CLS токен, позиционные и патч-эмбеддинги для данных WSI.
    """

    def __init__(self, cfg):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.patch_embeddings = WsiTMAEPatchEmbeddings(cfg)
        self.num_patches = self.patch_embeddings.num_patches
        # фиксированные синусо-косинусные эмбеддинги
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, cfg.hidden_size), requires_grad=False
        )
        self.patch_size = cfg.patch_size
        self.config = cfg
        self.initialize_weights()

    def initialize_weights(self):
        # Инициализируем (и замораживаем) позиционные эмбеддинги с помощью синусо-косинусных эмбеддингов
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.position_embeddings.shape[-1], 
            np.arange(int(self.patch_embeddings.num_patches), dtype=np.float32)
        )
        pos_embed = np.concatenate([
            np.zeros([1, self.position_embeddings.shape[-1]]),  # для CLS токена
            pos_embed
        ], axis=0)
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w)

        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    def random_masking(self, sequence, noise=None):
        """
        Выполняет случайное маскирование для каждого образца, перетасовывая элементы.
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)

        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)  # Перетасовка
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def forward(self, wsi_values, noise=None):
        batch_size, num_channels, img_size1, img_size2 = wsi_values.shape
        embeddings = self.patch_embeddings(wsi_values)
        
        embeddings = embeddings + self.position_embeddings[:, 1:, :]
        
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

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
        return rearrange(
            imgs, 
            'b c (h p1) (w p2) -> b (h w) (c p1 p2)', 
            p1=self.config.patch_size, 
            p2=self.config.patch_size
        )


class WsiMAEDecoder(nn.Module):
    def __init__(self, config, num_patches):
        super().__init__()
        self.decoder_pred = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, config.num_channels * config.img_size ** 2),
            Rearrange("b (c h w) -> b c h w", c=config.num_channels, h=config.img_size, w=config.img_size)
        )

    def forward(self, x):
        return self.decoder_pred(x)


class WsiMAEForPreTraining(ViTMAEForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vit = WsiMAEModel(config)
        self.decoder = WsiMAEDecoder(config, num_patches=self.vit.embeddings.num_patches)
        self.post_init()

    def patchify(self, imgs):
        return rearrange(
            imgs, 
            'b c (h p1) (w p2) -> b (h w) (c p1 p2)', 
            p1=self.config.patch_size, 
            p2=self.config.patch_size
        )


class WsiMaeSurvivalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.to_dict().get("is_load_pretrained", False):
            self.vit = WsiMAEModel.from_pretrained(config.pretrained_model_path, config=config)
        else:
            self.vit = WsiMAEModel(config)
        self.projection = nn.Linear(config.hidden_size, config.output_dim)
        
    def forward(self, wsi_values, masks=None):
        x = self.vit(wsi_values)
        x = self.projection(x.last_hidden_state[:, 0, :])
        return x.squeeze(-1)
