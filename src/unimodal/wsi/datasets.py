from typing import Tuple, Union, Any
from PIL import Image
import torch
import pandas as pd
from src.datasets import BaseDataset  # Обновлено на BaseDataset, согласно новым изменениям
import os
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F

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
        dataframe: pd.DataFrame,
        k: int,
        is_train: bool = True,
        return_mask: bool = False,
        transform: "torchvision.transforms" = None,
        is_hazard_logits = False # Добавлен параметр
    ) -> None:
        super().__init__(data=dataframe, transform=transform, return_mask=return_mask, is_hazard_logits=is_hazard_logits)
        self.k = k
        self.is_train = is_train

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, bool], torch.Tensor]:
        sample = self.data.iloc[idx]
        
        if not pd.isna(sample.WSI):
            data = pd.read_csv(sample.WSI)
            # get k random embeddings
            if self.is_train:
                data = data.sample(self.k)
            else:
                data = data.iloc[:self.k]

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

from torchvision.transforms import functional as F

class WSIDataset_patches(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        return_mask: bool = False,
        transform: "torchvision.transforms" = None,
        is_hazard_logits: bool = False,
        batch_size: int = 16,
        resize_to: tuple = (256, 256),
        patch_size: tuple = (64, 64),  # Новый параметр для размера подпатча
    ) -> None:
        self.data = dataframe
        self.transform = transform
        self.is_hazard_logits = is_hazard_logits
        self.batch_size = batch_size
        self.resize_to = resize_to
        self.patch_size = patch_size  # Сохраняем размер подпатча

    def _load_and_split_patches(self, image_dir: str):
        """Загружает патчи и делит их на подпатчи."""
        patch_dir = os.path.join(image_dir, "patches")
        patch_files = sorted(os.listdir(patch_dir))
        
        sub_patches = []
        for patch_file in patch_files:
            if patch_file.endswith(".png"):
                patch_path = os.path.join(patch_dir, patch_file)
                patch = Image.open(patch_path).convert("RGB")
                patch = F.resize(patch, self.resize_to)  # Изменяем размер основного патча
                
                # Разбиваем патч на подпатчи 64x64
                for i in range(0, patch.size[1], self.patch_size[1]):
                    for j in range(0, patch.size[0], self.patch_size[0]):
                        sub_patch = F.crop(patch, i, j, self.patch_size[1], self.patch_size[0])
                        sub_patches.append(sub_patch)
        
        return sub_patches

    def __getitem__(self, idx: int):
        sample = self.data.iloc[idx]
        
        if not pd.isna(sample.WSI_initial):
            svs_path = sample.WSI_initial
            image_dir = os.path.dirname(svs_path)
            sub_patches = self._load_and_split_patches(image_dir)
            
            if self.transform:
                sub_patches = [F.pil_to_tensor(self.transform(patch)) for patch in sub_patches]
            else:
                sub_patches = [F.pil_to_tensor(patch) for patch in sub_patches]
            
            sub_patches = [patch.float() for patch in sub_patches]
            sub_patches = torch.stack(sub_patches)  # Преобразуем в тензор
            
            # Отбрасываем лишние патчи
            num_full_batches = len(sub_patches) // self.batch_size
            sub_patches = sub_patches[:num_full_batches * self.batch_size]
            
            return sub_patches, True
        else:
            return torch.zeros((1, 3, *self.patch_size)), False
        

class SurvivalWSIDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        split: pd.DataFrame,  # DataFrame с временными и событийными данными
        dataset: torch.utils.data.Dataset,  
        is_hazard_logits: bool = False  # Параметр, который указывает, нужно ли использовать логиты
    ) -> None:
        self.dataset = dataset 
        
        # В зависимости от того, логиты ли это, или другие данные, мы подбираем колонки для времени и события
        if is_hazard_logits:
            self.time = torch.from_numpy(split["new_time"].values)
            self.event = torch.from_numpy(split["new_event"].values)
        else:
            self.time = torch.from_numpy(split["time"].values)
            self.event = torch.from_numpy(split["event"].values)

    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        # Возвращаем патч из датасета WSI, время и событие для выживания
        return self.dataset[idx], self.time[idx], self.event[idx]

    def __len__(self) -> int:
        return len(self.dataset)  # Размер датасета (количество патчей)
