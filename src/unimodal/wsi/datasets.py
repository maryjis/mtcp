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
from pathlib import Path

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
        random_patch_selection: bool = False,
        debug_mode =False,
        fill_missing ="zeros"
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
        self.fill_missing =fill_missing
        self.debug_mode =debug_mode
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
            if self.fill_missing =="zeros":
                return torch.zeros((expected_patches, 3, *self.resize_to), dtype=torch.float32), False
            elif self.fill_missing =="normal":
                return torch.randn((expected_patches, 3, *self.resize_to), dtype=torch.float32), False
            else:
                raise NotImplementedError
        try:
            patch_files = sorted([f for f in os.listdir(patch_dir) if f.endswith(".png")])
        except Exception as e:
            logger.exception(f"Failed to list files in {patch_dir}")
            if self.fill_missing =="zeros":
                return torch.zeros((expected_patches, 3, *self.resize_to), dtype=torch.float32) , False
            elif self.fill_missing =="normal":
                return torch.randn((expected_patches, 3, *self.resize_to), dtype=torch.float32), False
            else:
                raise NotImplementedError

        if not patch_files:
            logger.warning(f"No patch files found in {patch_dir}")
            if self.fill_missing =="zeros":
                return torch.zeros((expected_patches, 3, *self.resize_to), dtype=torch.float32) , False
            elif self.fill_missing =="normal":
                return torch.randn((expected_patches, 3, *self.resize_to), dtype=torch.float32), False
            else:
                raise NotImplementedError

        # Выбор одного случайного патча
        if self.random_patch_selection:
            if len(patch_files) == 0:
                logger.warning(f"No patch files to select randomly in {patch_dir}")
                return torch.zeros((1, 3, *self.resize_to), dtype=torch.float32), False
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
            if self.fill_missing =="zeros":
                return torch.zeros((expected_patches, 3, *self.resize_to), dtype=torch.float32) , False
            elif self.fill_missing =="normal":
                return torch.randn((expected_patches, 3, *self.resize_to), dtype=torch.float32), False
            else:
                raise NotImplementedError

        # Паддинг, если не хватает патчей
        while not self.random_patch_selection and len(patches) < self.max_patches_per_sample:
            
            if self.fill_missing =="zeros":
                patches.append(torch.zeros((3, *self.resize_to), dtype=torch.float32))
            elif self.fill_missing =="normal":
                patches.append(torch.randn((3, *self.resize_to), dtype=torch.float32))
            else:
                raise NotImplementedError

            logger.debug(f"Added zero-padding patch. Total now: {len(patches)}")

        tensor = torch.stack(patches)

        # Финальная проверка
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logger.warning(f"NaN or Inf detected in tensor from {patch_dir}")

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(f"Loaded {len(patches)} patches from {patch_dir} in {elapsed:.1f} ms. Shape: {tensor.shape}")

        return tensor, True

    def __getitem__(self, idx: int):
        t0 = time.perf_counter()
        sample = self.data.iloc[idx]
        patch_dir = sample.WSI_initial if not pd.isna(sample.WSI_initial) else None

        if patch_dir:
            patches,mask = self._load_patches(patch_dir)
        else:
            patches = torch.zeros(
                (1 if self.random_patch_selection else self.max_patches_per_sample, 3, *self.resize_to),
                dtype=torch.float32
            )
            mask = False
            logger.warning(f"Sample at idx={idx} has no WSI_initial. Returning zero patches.")

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(f"__getitem__ idx={idx} | path={patch_dir} | shape={patches.shape} | time={elapsed:.1f} ms")
        if not self.debug_mode:
            return (patches, mask) if self.return_mask else patches
        else:
            return (str(patch_dir), patches, mask) if self.return_mask else (str(patch_dir),patches)

    def __len__(self):
        return len(self.data)

class WSIDataset_embeddings(BaseDataset):
    def __init__(
        self,
        data: pd.DataFrame,
        transform: "torchvision.transforms" = None,
        return_mask: bool = False,
        is_hazard_logits: bool = False,
        debug_mode =False,
        fill_missing ="zeros",
        embedding_dim =768,
        embedding_pattern: str = "*_gigapath.pt",
        embedding_key: str = "last_layer_embed",
    ) -> None:
        super().__init__(
            data=data,
            transform=transform,
            return_mask=return_mask,
            is_hazard_logits=is_hazard_logits
        )
        self.embedding_dim = embedding_dim
        self.debug_mode = debug_mode
        self.embedding_pattern = embedding_pattern
        self.embedding_key = embedding_key
        
    def find_embedding_checkpoint(self, folder: str | Path):
        folder = Path(folder)
        matches = sorted(folder.glob(self.embedding_pattern))
        if not matches:
            logger.warning(f"Not found '{self.embedding_pattern}' in folder: {folder}, Returning zero embedding.")
            return None, False
        if len(matches) > 1:
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0], True
    
    def __getitem__(self, idx: int):
        sample = self.data.iloc[idx]
        patch_dir = sample.WSI_initial if not pd.isna(sample.WSI_initial) else None
        wsi_dir  = patch_dir[:-7] if patch_dir else None
        if wsi_dir:
            ckpt_path, mask = self.find_embedding_checkpoint(wsi_dir)
            try:
                wsi_embed_full = torch.load(ckpt_path, map_location="cpu")
                wsi_embed = wsi_embed_full[self.embedding_key]
            except Exception as e:
                logger.warning(f"Can't load embedding: {e}")
                wsi_embed = torch.zeros(self.embedding_dim, dtype=torch.float32)
                mask = False
        else:
            wsi_embed = torch.zeros(self.embedding_dim, dtype=torch.float32)
            mask = False
            logger.warning(f"Sample at idx={idx} has no WSI embedding. Returning zero embedding.")

        if not self.debug_mode:
            return (wsi_embed, mask) if self.return_mask else wsi_embed
        else:
            return (str(patch_dir), wsi_embed, mask) if self.return_mask else (str(patch_dir),wsi_embed)

        


class WSIDataset_patch_features(BaseDataset):
    """
    Loads patch-level features (e.g. CONCHv1.5) and coordinates from h5 files
    for use with the live TITAN encoder.

    Returns a packed tensor [max_patches, feature_dim + 2]:
      - [:, :feature_dim]  — patch features (768-d by default)
      - [:, feature_dim:]  — patch coordinates (x, y)
    Zero-padded for slides with fewer than max_patches patches.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        transform: "torchvision.transforms" = None,
        return_mask: bool = False,
        is_hazard_logits: bool = False,
        debug_mode: bool = False,
        fill_missing: str = "zeros",
        feature_dim: int = 768,
        max_patches: int = 4096,
        h5_feature_key: str = "features",
        h5_coord_key: str = "coords",
        h5_glob: str = "*.h5",
    ) -> None:
        super().__init__(
            data=data,
            transform=transform,
            return_mask=return_mask,
            is_hazard_logits=is_hazard_logits,
        )
        self.feature_dim = feature_dim
        self.max_patches = max_patches
        self.h5_feature_key = h5_feature_key
        self.h5_coord_key = h5_coord_key
        self.h5_glob = h5_glob
        self.debug_mode = debug_mode
        self.fill_missing = fill_missing
        self.packed_dim = feature_dim + 2

    def _find_h5(self, folder):
        folder = Path(folder)
        matches = sorted(folder.glob(self.h5_glob))
        return matches[0] if matches else None

    def __getitem__(self, idx: int):
        import h5py

        sample = self.data.iloc[idx]
        patch_dir = sample.WSI_initial if not pd.isna(sample.WSI_initial) else None
        wsi_dir = patch_dir[:-7] if patch_dir else None

        packed = torch.zeros(self.max_patches, self.packed_dim, dtype=torch.float32)
        mask = False

        if wsi_dir:
            h5_path = self._find_h5(wsi_dir)
            if h5_path:
                try:
                    with h5py.File(str(h5_path), "r") as f:
                        feats = torch.from_numpy(f[self.h5_feature_key][:]).float()
                        coords = torch.from_numpy(f[self.h5_coord_key][:]).float()

                    N = feats.shape[0]
                    if N > self.max_patches:
                        sel = torch.randperm(N)[: self.max_patches]
                        feats = feats[sel]
                        coords = coords[sel]
                        N = self.max_patches

                    packed[:N, : self.feature_dim] = feats
                    packed[:N, self.feature_dim :] = coords[:, :2]
                    mask = True
                except Exception as e:
                    logger.warning("Failed to load h5 %s: %s", h5_path, e)

        if not self.debug_mode:
            return (packed, mask) if self.return_mask else packed
        return (str(patch_dir), packed, mask) if self.return_mask else (str(patch_dir), packed)


class SurvivalWSIDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        split: pd.DataFrame,
        dataset: torch.utils.data.Dataset,
        is_hazard_logits: bool = False,
        debug_mode =False
    ) -> None:
        self.dataset = dataset
        self.data = self.dataset.data
        self.debug_mode = debug_mode
        if is_hazard_logits:
            self.time = torch.from_numpy(split["new_time"].values)
            self.event = torch.from_numpy(split["new_event"].values)
        else:
            self.time = torch.from_numpy(split["time"].values)
            self.event = torch.from_numpy(split["event"].values)

    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        if not self.debug_mode:
            data, mask = self.dataset[idx]  # Разбиваем tuple
            return data, mask, self.time[idx], self.event[idx]
        else: 
            file_ids, data, mask = self.dataset[idx]
            return file_ids, data, mask, self.time[idx], self.event[idx]

        #print(f"Index {idx} -> Data shape: {data.shape},  Time: {self.time[idx]}, Event: {self.event[idx]}")
        
    def __len__(self) -> int:
        return len(self.dataset)



