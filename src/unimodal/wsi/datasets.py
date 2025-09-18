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
import time
import logging
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as F

logger = logging.getLogger(__name__)

class WSIDataset_patches(BaseDataset):
    def __init__(
        self,
        data: pd.DataFrame,
        transform: "torchvision.transforms" = None,
        return_mask: bool = False,
        is_hazard_logits: bool = False,
        resize_to: tuple = (224, 224),
        max_patches_per_sample: int = 10,
        random_patch_selection: bool = False
    ) -> None:
        super().__init__(
            data=data,
            transform=transform,
            return_mask=return_mask,
            is_hazard_logits=is_hazard_logits
        )
        self.resize_to = resize_to
        self.max_patches_per_sample = max_patches_per_sample
        self.random_patch_selection = random_patch_selection

        logger.info(
            f"WSIDataset_patches initialized: n_samples={len(self.data)}, "
            f"resize_to={self.resize_to}, max_patches_per_sample={self.max_patches_per_sample}, "
            f"random_patch_selection={self.random_patch_selection}, return_mask={self.return_mask}"
        )
        if "WSI_initial" in self.data.columns:
            missing = self.data["WSI_initial"].isna().sum()
            logger.info(f"Missing WSI_initial paths: {missing} of {len(self.data)}")

    def _load_patches(self, patch_dir: str) -> torch.Tensor:
        """Загружает патчи из директории `patch_dir`, масштабирует и конвертирует в тензоры"""
        t0 = time.perf_counter()
        expected_patches = 1 if self.random_patch_selection else self.max_patches_per_sample

        if not os.path.exists(patch_dir):
            logger.warning(f"Patch directory not found: {patch_dir}")
            return torch.zeros((expected_patches, 3, *self.resize_to), dtype=torch.float32)

        try:
            patch_files = sorted([f for f in os.listdir(patch_dir) if f.endswith(".png")])
        except Exception as e:
            logger.exception(f"Failed to list files in {patch_dir}")
            return torch.zeros((expected_patches, 3, *self.resize_to), dtype=torch.float32)

        if not patch_files:
            logger.warning(f"No patch files found in {patch_dir}")
            return torch.zeros((expected_patches, 3, *self.resize_to), dtype=torch.float32)

        # Выбор одного случайного патча
        if self.random_patch_selection:
            if len(patch_files) == 0:
                logger.warning(f"No patch files to select randomly in {patch_dir}")
                return torch.zeros((1, 3, *self.resize_to), dtype=torch.float32)
            idx_patch = torch.randint(0, len(patch_files), (1,)).item()
            patch_files = [patch_files[idx_patch]]
            logger.debug(f"Randomly selected patch: {patch_files[0]} from {patch_dir}")

        patches = []
        for patch_file in patch_files[:self.max_patches_per_sample]:
            patch_path = os.path.join(patch_dir, patch_file)
            try:
                patch = Image.open(patch_path).convert("RGB")
                patch = F.resize(patch, self.resize_to)
                patch = F.pil_to_tensor(patch).float()

                # Нормализация
                patch = torch.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)
                min_val, max_val = patch.min(), patch.max()
                if (max_val - min_val).abs() < 1e-8:
                    logger.warning(f"Flat patch (zero variance) in {patch_path}")
                    patch = torch.zeros_like(patch)
                else:
                    patch = (patch - min_val) / (max_val - min_val + 1e-8)
          
                # Применение transform
                if self.transform:
                    patch = self.transform(patch)

                patches.append(patch)
            except Exception:
                logger.exception(f"Failed to load or process patch: {patch_path}")
                continue

        if not patches:
            logger.warning(f"All patch loads failed for {patch_dir}, returning zeros.")
            return torch.zeros((expected_patches, 3, *self.resize_to), dtype=torch.float32)

        # Паддинг, если не хватает патчей
        while not self.random_patch_selection and len(patches) < self.max_patches_per_sample:
            patches.append(torch.zeros((3, *self.resize_to), dtype=torch.float32))
            logger.debug(f"Added zero-padding patch. Total now: {len(patches)}")

        tensor = torch.stack(patches)

        # Финальная проверка
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logger.warning(f"NaN or Inf detected in tensor from {patch_dir}")

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(f"Loaded {len(patches)} patches from {patch_dir} in {elapsed:.1f} ms. Shape: {tensor.shape}")

        return tensor

    def __getitem__(self, idx: int):
        t0 = time.perf_counter()
        sample = self.data.iloc[idx]
        patch_dir = sample.WSI_initial if not pd.isna(sample.WSI_initial) else None
        patch_dir ="/data/TCGA_all_wsi/" +patch_dir +"/patches"
        if patch_dir:
            patches = self._load_patches(patch_dir)
            mask = True
        else:
            patches = torch.zeros(
                (1 if self.random_patch_selection else self.max_patches_per_sample, 3, *self.resize_to),
                dtype=torch.float32
            )
            mask = False
            logger.warning(f"Sample at idx={idx} has no WSI_initial. Returning zero patches.")

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(f"__getitem__ idx={idx} | path={patch_dir} | shape={patches.shape} | time={elapsed:.1f} ms")

        return (patches, mask) if self.return_mask else patches

    def __len__(self):
        return len(self.data)


        


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



