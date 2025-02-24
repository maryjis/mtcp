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

###############################
# Класс для разбиения и эмбеддингов подпатчей
###############################
class WsiMAEPatchEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.image_size = cfg.image_size       # например, 256
        self.patch_size = cfg.patch_size       # например, 16
        self.hidden_size = cfg.hidden_size
        self.num_channels = cfg.num_channels
        max_patches_per_sample = cfg.max_patches_per_sample
        
        # Количество subpatchей (подпатчей) в одном большом патче:
        self.num_sub_patches = (self.image_size // self.patch_size) ** 2
        # Общее число токенов если бы все большие патчи обрабатывались вместе:
        self.total_patches = max_patches_per_sample * self.num_sub_patches

        # Здесь можно оставить ту же сверточную "голову" для каждого подпатча:
        layers = []
        in_ch = self.num_channels
        out_ch = 64  # начальное число каналов
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(out_ch, self.hidden_size))
        self.projection = nn.Sequential(*layers)

    def _split_patches(self, wsi_tensor):
        """
        Разбивает один большой патч (вход с формой [B', C, image_size, image_size])
        на subpatchи размером patch_size x patch_size.
        Результат имеет форму [B', num_sub_patches, C, patch_size, patch_size].
        """
        b, c, h, w = wsi_tensor.shape
        p = self.patch_size
        if h % p != 0 or w % p != 0:
            raise ValueError(f"Размер изображения ({h}, {w}) не делится на ({p}, {p}) без остатка.")
        patches = wsi_tensor.unfold(2, p, p).unfold(3, p, p)  # [b, c, h//p, p, w//p, p]
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()  # [b, h//p, w//p, c, p, p]
        patches = patches.view(b, (h // p) * (w // p), c, p, p)
        return patches

    def forward(self, wsi_tensor):
        """
        Входной тензор имеет форму:
          [batch, max_patches_per_sample, channels, image_size, image_size]
        Переразбиваем данные так, чтобы каждый большой патч стал отдельным элементом батча.
        Результат: [batch * max_patches_per_sample, num_sub_patches, hidden_size]
        """
        b, m, c, h, w = wsi_tensor.shape
        # Каждый большой патч отдельно:
        wsi_tensor = wsi_tensor.view(b * m, c, h, w)  # [b*m, c, h, w]
        patches = self._split_patches(wsi_tensor)       # [b*m, num_sub_patches, c, patch_size, patch_size]
        patches = rearrange(patches, 'B N c h w -> (B N) c h w')
        x = self.projection(patches)                      # [(b*m * num_sub_patches), hidden_size]
        x = rearrange(x, '(B N) c -> B N c', B=b * m, N=self.num_sub_patches)
        return x  # форма: [b*m, num_sub_patches, hidden_size]

###############################
# Класс эмбеддингов с позициональными и CLS-токенами
###############################
class WsiMAEEmbeddings(nn.Module):
    """
    Формирует эмбеддинги для каждого большого патча отдельно.
    Вход: [batch, max_patches_per_sample, channels, image_size, image_size]
    Выход: [batch * max_patches_per_sample, 1 + num_sub_patches, hidden_size]
    """
    def __init__(self, cfg):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        # Используем наш WsiMAEPatchEmbeddings, который уже переразбивает данные:
        self.patch_embeddings = WsiMAEPatchEmbeddings(cfg)
        # Число токенов для одного большого патча = число subpatchей:
        self.num_patches = self.patch_embeddings.num_sub_patches  
        # Позиционные эмбеддинги рассчитываются для (CLS + subpatchи):
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, cfg.hidden_size), requires_grad=False
        )
        self.patch_size = cfg.patch_size
        self.config = cfg
        self.initialize_weights()

    def initialize_weights(self):
        # Вычисляем синус-косин позиционные эмбеддинги для self.num_patches токенов
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.position_embeddings.shape[-1],
            np.arange(int(self.num_patches), dtype=np.float32)
        )
        pos_embed = np.concatenate([
            np.zeros([1, self.position_embeddings.shape[-1]]),  # для CLS токена
            pos_embed
        ], axis=0)
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
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
        # Получаем эмбеддинги для каждого большого патча
        # Вход: [B, N, C, H, W] → выход: [B * N, num_subpatches, hidden_size]
        embeddings = self.patch_embeddings(image_values)
        # Добавляем позиционные эмбеддинги (срез для одного большого патча)
        embeddings = embeddings + self.position_embeddings[:, 1:1+self.num_patches, :]
        # Маскирование (работает независимо для каждого элемента батча)
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)
        # Добавляем CLS-токен для каждого большого патча
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        # Выход: [B * N, 1 + num_subpatches, hidden_size]
        return embeddings, mask, ids_restore

###############################
# Модель ViTMAE с нашей реализацией embeddings
###############################
class WsiMAEModel(ViTMAEModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = WsiMAEEmbeddings(config)
        self.post_init()

    def patchify(self, imgs, interpolate_pos_encoding: bool = False):
        # Если требуется использовать patchify для расчёта целевых патчей,
        # можно передать входной тензор в виде [B, N, C, H, W],
        # затем перевести каждый большой патч в отдельный элемент батча.
        p = self.config.patch_size
        b, n, c, h, w = imgs.shape
        assert h == w == self.config.image_size, "Размер изображения не совпадает с config.image_size"
        assert h % p == 0 and w % p == 0, "Размер изображения должен делиться на patch_size"
        imgs = imgs.view(b * n, c, h, w)
        patches = rearrange(imgs, 'B c (h p1) (w p2) -> B (h w) c p1 p2', p1=p, p2=p)
        patches = patches.flatten(2)
        return patches

###############################
# Декодер. Он работает с входом вида [B_new, s, decoder_hidden_size],
# где B_new = B * N (каждый большой патч – отдельный элемент батча)
###############################
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
        # x имеет форму [B_new, s, decoder_hidden_size]
        B_new = x.shape[0]
        x = x.unsqueeze(-1).unsqueeze(-1)  # [B_new, s, decoder_hidden_size, 1, 1]
        x = rearrange(x, 'B_new s e x y -> (B_new s) e x y')
        x = self.projector(x)  # [(B_new * s), num_channels, patch_size, patch_size]
        x = rearrange(x, '(B_new s) c h w -> B_new s (c h w)', B_new=B_new)
        return x

class WsiMAEDecoder(ViTMAEDecoder):
    def __init__(self, config, num_patches):
        super().__init__(config, num_patches)
        self.decoder_pred = WsiMAEDecoderPred(config)
        self.initialize_weights(num_patches)
        
    def initialize_weights(self, num_patches):
        decoder_pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.decoder_pos_embed.shape[-1],
            np.arange(int(num_patches), dtype=np.float32)
        )
        decoder_pos_embed = np.concatenate([
            np.zeros([1, self.decoder_pos_embed.shape[-1]]),
            decoder_pos_embed
        ], axis=0)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

###############################
# Предобучение MAE с нашей реализацией
###############################
class WsiMAEForPreTraining(ViTMAEForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vit = WsiMAEModel(config)
        # Передаём число токенов (subpatchей) для одного большого патча
        self.decoder = WsiMAEDecoder(config, num_patches=self.vit.embeddings.num_patches)
        self.post_init()

    def forward(self, image_values, **kwargs):
        """
        Из входного батча [B, max_patches_per_sample, C, H, W]
        случайным образом выбирается по одному большому патчу для каждого образца.
        Далее происходит стандартный MAE-процесс.
        """
        B, N, C, H, W = image_values.shape
        # Случайно выбираем один патч для каждого сэмпла
        idx = torch.randint(0, N, (B,), device=image_values.device)
        selected = image_values[torch.arange(B), idx]  # [B, C, H, W]
        # Добавляем размерность, чтобы получить форму [B, 1, C, H, W]
        selected = selected.unsqueeze(1)
        return super().forward(selected, **kwargs)

    def patchify(self, imgs, interpolate_pos_encoding: bool = False):
        # Переопределяем patchify для обработки [B, 1, C, H, W]
        p = self.config.patch_size
        b, n, c, h, w = imgs.shape
        assert h == w == self.config.image_size, "Размер изображения не совпадает с config.image_size"
        assert h % p == 0 and w % p == 0, "Размер изображения должен делиться на patch_size"
        imgs = imgs.view(b * n, c, h, w)
        patches = rearrange(imgs, 'B c (h p1) (w p2) -> B (h w) c p1 p2', p1=p, p2=p)
        patches = patches.flatten(2)
        return patches

###############################
# Survival-модель, агрегирующая результаты по пациентам
###############################
class WsiMaeSurvivalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Загрузка предобученной модели или инициализация с нуля
        if config.to_dict().get("is_load_pretrained", False):
            self.vit = WsiMAEModel.from_pretrained(config.pretrained_model_path, config=config)
            print(f"Pretrained model loaded from {config.pretrained_model_path}")
        else:
            self.vit = WsiMAEModel(config)
        self.projection = nn.Linear(config.hidden_size, config.output_dim)
        # Сохраняем число больших патчей на сэмпл
        self.max_patches_per_sample = config.max_patches_per_sample

    def forward(self, wsi_values, masks=None):
        # wsi_values имеет форму [B, max_patches_per_sample, C, H, W]
        vit_out = self.vit(wsi_values)
        # Предполагаем, что vit_out.last_hidden_state имеет форму:
        # [B_new, tokens, hidden_size], где B_new = B * max_patches_per_sample
        # CLS-токен находится на позиции 0 для каждого большого патча.
        cls_tokens = vit_out.last_hidden_state[:, 0, :]  # [B_new, hidden_size]
        # Восстанавливаем исходное распределение:
        N = self.max_patches_per_sample
        B_new = cls_tokens.shape[0]
        B = B_new // N
        cls_tokens = cls_tokens.view(B, N, -1)  # [B, N, hidden_size]
        # Агрегируем информацию для каждого пациента (усредняем по N больших патчей)
        patient_repr = cls_tokens.mean(dim=1)  # [B, hidden_size]
        x = self.projection(patient_repr)
        return x.squeeze(-1)

