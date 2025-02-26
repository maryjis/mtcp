from typing import Tuple, Union, Any
from PIL import Image
import torch
import pandas as pd
from src.datasets import BaseDataset  # Обновлено на BaseDataset, согласно новым изменениям
import os
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils.data import Dataset


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, filepaths: Tuple[str, ...], transform: "torchvision.transforms" = None) -> None:
        self.filepaths = filepaths
        self.transform = transform  # Параметр должен быть в единственном числе

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.filepaths[idx]
        image = Image.open(path)
        if self.transform:
            image_1, image_2 = self.transform(image)
        else:
            image_1, image_2 = image, image  # Если трансформации нет, возвращаем само изображение
        return image_1, image_2

    def __len__(self) -> int:
        return len(self.filepaths)


class WSIDataset(BaseDataset):
    def __init__(
        self,
        data: pd.DataFrame,
        k: int,
        is_train: bool = True,
        return_mask: bool = False,
        transform: "torchvision.transforms" = None,
        is_hazard_logits = False # Добавлен параметр
    ) -> None:
        super().__init__(data=data, transform=transform, return_mask=return_mask, is_hazard_logits=is_hazard_logits)
        self.k = k
        self.is_train = is_train

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, bool], torch.Tensor]:
        sample = self.data.iloc[idx]
        
        if not pd.isna(sample.WSI):
            data = pd.read_csv(sample.WSI)
            # get k random embeddings
            data = data.sample(self.k)  

            data = torch.from_numpy(data.values).float()
            mask = True
        else:
            data = torch.zeros(self.k, 512).float()
            mask = False

        if self.transform:
            data = self.transform(data)

        if self.return_mask:
            return data, mask
        else:
            return data



import os
import torch
import pandas as pd
from torchvision.transforms import functional as F
from PIL import Image

class WSIDataset_patches(BaseDataset):
    def __init__(
        self,
        data: pd.DataFrame,
        transform: "torchvision.transforms" = None,
        return_mask: bool = False,
        is_hazard_logits=False,
        resize_to: tuple = (256, 256),
        max_patches_per_sample: int = 10,
        random_patch_selection: bool = False  # добавленный параметр
    ) -> None:
        super().__init__(data=data, transform=transform, return_mask=return_mask, is_hazard_logits=is_hazard_logits)
        self.resize_to = resize_to
        self.max_patches_per_sample = max_patches_per_sample
        self.random_patch_selection = random_patch_selection

    def _load_patches(self, image_dir: str):
        """Загружает патчи, масштабирует и конвертирует в тензоры"""
        patch_dir = os.path.join(image_dir, "patches")

        # Если папки нет, вернуть нужное количество пустых патчей
        if not os.path.exists(patch_dir):
            if self.random_patch_selection:
                return torch.zeros((1, 3, *self.resize_to), dtype=torch.float32)  # Один нулевой патч
            else:
                return torch.zeros((self.max_patches_per_sample, 3, *self.resize_to), dtype=torch.float32)  # `max_patches_per_sample` нулевых патчей

        patch_files = sorted([f for f in os.listdir(patch_dir) if f.endswith(".png")])

        # Если папка есть, но пустая, возвращаем нужный размер нулевых патчей
        if not patch_files:
            if self.random_patch_selection:
                return torch.zeros((1, 3, *self.resize_to), dtype=torch.float32)  # Один нулевой патч
            else:
                return torch.zeros((self.max_patches_per_sample, 3, *self.resize_to), dtype=torch.float32)  # `max_patches_per_sample` нулевых патчей

        patches = []

        if self.random_patch_selection:
            # Выбираем один случайный патч
            max_idx = min(len(patch_files), self.max_patches_per_sample)
            idx_patch = torch.randint(0, max_idx, (1,)).item()
            patch_files = [patch_files[idx_patch]]

        for patch_file in patch_files[:self.max_patches_per_sample]:  # Ограничение по `max_patches_per_sample`
            patch_path = os.path.join(patch_dir, patch_file)
            patch = Image.open(patch_path).convert("RGB")
            patch = F.resize(patch, self.resize_to)
            patch = F.pil_to_tensor(patch).float() / 255.0
            patches.append(patch)

        if self.random_patch_selection:
            # Гарантируем, что размер всегда `(1, 3, 256, 256)`
            patches = torch.stack(patches) if patches else torch.zeros((1, 3, *self.resize_to), dtype=torch.float32)
        else:
            # Гарантируем, что размер всегда `(max_patches_per_sample, 3, 256, 256)`
            while len(patches) < self.max_patches_per_sample:
                patches.append(torch.zeros((3, *self.resize_to), dtype=torch.float32))

        patches = torch.stack(patches)

        return patches  # Теперь `torch.Tensor` с гарантированным размером



    def __getitem__(self, idx: int):
        sample = self.data.iloc[idx]

        if not pd.isna(sample.WSI_initial):
            svs_path = sample.WSI_initial
            image_dir = os.path.dirname(svs_path)
            patches = self._load_patches(image_dir)  # `_load_patches()` уже возвращает `torch.Tensor`
            mask = True
        else:
            patches = torch.zeros(
                (1 if self.random_patch_selection else self.max_patches_per_sample, 3, *self.resize_to), 
                dtype=torch.float32)
            mask = False

        if self.return_mask:
            return patches, mask
        else:
            return patches
  
    def __len__(self):
        return len(self.data)  # Теперь длина = числу WSI, а не батчей!


        


class SurvivalWSIDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        split: pd.DataFrame,
        dataset: torch.utils.data.Dataset,
        is_hazard_logits: bool = False,
    ) -> None:
        self.dataset = dataset
        if is_hazard_logits:
            self.time = torch.from_numpy(split["new_time"].values)
            self.event = torch.from_numpy(split["new_event"].values)
        else:
            self.time = torch.from_numpy(split["time"].values)
            self.event = torch.from_numpy(split["event"].values)

    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        data, mask = self.dataset[idx]  # Разбиваем tuple
        #print(f"Index {idx} -> Data shape: {data.shape},  Time: {self.time[idx]}, Event: {self.event[idx]}")
        return data, mask, self.time[idx], self.event[idx]

    def __len__(self) -> int:
        return len(self.dataset)
