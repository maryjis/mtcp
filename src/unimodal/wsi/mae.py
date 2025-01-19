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

class WsiMAEPatchEmbeddings(nn.Module):
    """
    This class processes 2D MRI patches already stored in PNG format for each image,
    turning them into the initial hidden states (patch embeddings) to be consumed by a Transformer.
    """

    def __init__(self, cfg):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.hidden_size = cfg.hidden_size
        self.num_channels = cfg.num_channels
        
        # 2D convolution to extract features from patches
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def _load_patches(self, image_dir):
        """Загружаем все патчи для изображения из соответствующей папки."""
        patch_files = sorted(os.listdir(image_dir))
        patches = []

        for patch_file in patch_files:
            if patch_file.endswith(".png"):
                patch_path = os.path.join(image_dir, patch_file)
                patch = Image.open(patch_path).convert("RGB")
                patch = transforms.ToTensor()(patch)  # Преобразуем изображение в тензор
                patches.append(patch)

        return torch.stack(patches)  # Возвращаем все патчи как один тензор

    def forward(self, patches):
        """
        Передаем патчи, полученные из WSIDataset_patches, через модель для извлечения эмбеддингов.
        patches: Размер: (num_patches, channels, patch_size, patch_size)
        """
        # Применяем свертку для извлечения эмбеддингов
        x = self.projection(patches)  # Применяем 2D свертку

        # Результат свертки будет иметь форму (num_patches, hidden_size, 1, 1)
        x = rearrange(x, 'n c h w -> n c')  # Преобразуем в (num_patches, hidden_size)
        
        return x

class WsiMAEEmbeddings(nn.Module):
    """
    Конструирует CLS токен, позиционные и патч-эмбеддинги для данных WSI.
    """

    def __init__(self, cfg):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.patch_embeddings = WsiMAEPatchEmbeddings(cfg)
        self.num_patches = self.patch_embeddings.num_patches

        # Фиксированные синусо-косинусные эмбеддинги
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, cfg.hidden_size), requires_grad=False
        )
        self.patch_size = cfg.patch_size
        self.config = cfg
        self.initialize_weights()

    def initialize_weights(self):
        # Инициализация позиционных эмбеддингов с помощью синусо-косинусных значений
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.position_embeddings.shape[-1], 
            np.arange(int(self.patch_embeddings.num_patches), dtype=np.float32)
        )
        pos_embed = np.concatenate([
            np.zeros([1, self.position_embeddings.shape[-1]]),  # Для CLS токена
            pos_embed
        ], axis=0)
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Инициализация весов свертки
        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w)

        # Инициализация CLS токена
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
        # Получаем эмбеддинги патчей
        embeddings = self.patch_embeddings(wsi_values)
        
        # Добавляем позиционные эмбеддинги
        embeddings = embeddings + self.position_embeddings[:, 1:, :]
        
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        # Добавляем CLS токен
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
        # Используем линейный слой для восстановления исходных данных
        self.decoder_pred = nn.Sequential(
            nn.Linear(config.hidden_size, 256),  # Преобразуем размерность
            nn.BatchNorm1d(256),  # Нормализация
            nn.ReLU(),  # Функция активации
            nn.Linear(256, config.num_channels * config.img_size ** 2),  # Преобразуем обратно к исходному размеру
            rearrange("b (c h w) -> b c h w", c=config.num_channels, h=config.img_size, w=config.img_size)  # Восстановление 2D изображения
        )

    def forward(self, x):
        return self.decoder_pred(x)

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
        """
        Преобразует изображения в патчи для подачи в трансформер.
        """
        return rearrange(
            imgs, 
            'b c (h p1) (w p2) -> b (h w) (c p1 p2)',  # Разбиваем изображение на патчи
            p1=self.config.patch_size, 
            p2=self.config.patch_size
        )

    def forward(self, wsi_values):
        """
        Основной forward проход для предобучения модели MAE.
        """
        # Получаем эмбеддинги с помощью ViT
        embeddings, mask, ids_restore = self.vit(wsi_values)

        # Применяем декодер для восстановления данных
        reconstructed = self.decoder(embeddings)

        return reconstructed, mask, ids_restore

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


