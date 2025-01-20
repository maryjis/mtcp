from typing import Tuple, Union, Any
from PIL import Image
import torch
import pandas as pd
from src.datasets import BaseDataset  # Обновлено на BaseDataset, согласно новым изменениям
import os
import numpy as np
from torchvision import transforms

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

class WSIDataset_patches(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        return_mask: bool = False,
        transform: "torchvision.transforms" = None,
        is_hazard_logits: bool = False,  # Добавлен параметр
        batch_size: int = 16,  # Новый параметр для размера батча (количество папок)
        resize_to: tuple = (256, 256)  # Размер, до которого нужно привести все патчи
    ) -> None:
        self.data = dataframe
        self.transform = transform
        self.is_hazard_logits = is_hazard_logits
        self.batch_size = batch_size  # Сохраняем batch_size (количество папок)
        self.resize_to = resize_to  # Размер для изменения всех патчей

    def _load_patches(self, image_dir: str):
        """Загружает патчи из папки `patches` для одного изображения"""
        patch_dir = os.path.join(image_dir, "patches")  # Папка с патчами
        patch_files = sorted(os.listdir(patch_dir))
        
        patches = []
        for patch_file in patch_files:
            if patch_file.endswith(".png"):
                patch_path = os.path.join(patch_dir, patch_file)
                patch = Image.open(patch_path).convert("RGB")
                patches.append(patch)

        return patches  # Возвращаем список изображений

    def __getitem__(self, idx: int):
        sample = self.data.iloc[idx]
        
        if not pd.isna(sample.WSI):
            # Извлекаем путь до папки, где находится .svs файл
            svs_path = sample.WSI
            image_dir = os.path.dirname(svs_path)  # Папка, содержащая .svs файл

            # Загружаем патчи из папки "patches"
            patches = self._load_patches(image_dir)
            mask = True
            print(type(patches))
            
            # Преобразуем патчи в тензоры с применением трансформаций
            if self.transform:
                patches = [transforms.functional.pil_to_tensor(self.transform(patch)) for patch in patches]  # Применяем трансформацию
            else:
                patches = [transforms.functional.pil_to_tensor(patch) for patch in patches]  # В случае, если это уже тензор

            # Приводим все патчи к одному размеру
            resize_transform = transforms.Resize(self.resize_to)  # Устанавливаем трансформацию на resize
            patches = [resize_transform(patch) for patch in patches]  # Применяем resize

            # Преобразуем каждый патч в тензор (если они не тензоры)
            patches = [patch.float() if isinstance(patch, torch.Tensor) else transforms.functional.pil_to_tensor(patch).float() for patch in patches] 

            # Преобразуем список патчей в тензор
            patches = torch.stack(patches)  # Преобразуем в тензор

            return patches, mask
        else:
            # Если данных нет, возвращаем пустой тензор
            return torch.zeros((1, 3, 224, 224)), False
        

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
