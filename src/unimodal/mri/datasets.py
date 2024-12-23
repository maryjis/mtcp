from typing import Optional, Union, List, Dict, Tuple, Any
import os
import numpy as np
import nibabel as nib
from abc import abstractmethod
from torch.utils.data import Dataset
import monai
import torch
import pandas as pd

from src.datasets import BaseDataset

__all__ = ["DatasetBraTSTumorCentered"]


class _BaseDatasetBraTS(Dataset):
    def __init__(
        self,
        path: str,
        modality: Union[str, List[str]],
        patients: Optional[List[str]],
        return_mask: bool,
        transform: Optional["monai.transforms"] = None,
    ) -> None:
        self.path = path
        self.modality = modality
        self.patients = np.array(os.listdir(path)) if patients is None else patients
        self.transform = transform
        self.return_mask = return_mask

    def __len__(self):
        return len(self.patients)

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    def _load_nifti_modalities(self, patient: str) -> np.ndarray:
        if len(self.modality) == 1:
            img = nib.load(
                os.path.join(
                    self.path, patient, patient + "-" + self.modality[0] + ".nii.gz"
                )
            ).get_fdata()
            img = np.expand_dims(img, 0)

        else:
            early_fused = []
            for modality in self.modality:
                early_fused.append(
                    nib.load(
                        os.path.join(
                            self.path, patient, patient + "-" + modality + ".nii.gz"
                        )
                    ).get_fdata()
                )

            img = np.stack(early_fused)

        return img

    @staticmethod
    def _monaify(img: np.ndarray, mask: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        item = {"image": img}
        if mask is not None:
            item["mask"] = mask

        return item

    def _load_mask(self, patient: str) -> np.ndarray:
        return nib.load(
            os.path.join(self.path, patient, patient + "-seg.nii.gz")
        ).get_fdata()


class DatasetBraTSTumorCentered(_BaseDatasetBraTS):
    def __init__(
        self,
        path: str,
        modality: Union[str, List[str]],
        patients: Optional[List[str]],
        sizes: Tuple[int, ...],
        transform: Optional["monai.transforms"] = None,
        return_mask: bool = False,
    ) -> None:
        super(DatasetBraTSTumorCentered, self).__init__(
            path=path,
            modality=modality,
            patients=patients,
            return_mask=return_mask,
            transform=transform,
        )
        self.sizes = sizes

    def __getitem__(self, idx) -> Dict[str, Union[torch.tensor, np.ndarray]]:
        patient = self.patients[idx]
        mask = super(DatasetBraTSTumorCentered, self)._load_mask(patient=patient)
        img = super(DatasetBraTSTumorCentered, self)._load_nifti_modalities(
            patient=patient
        )
        item = self._compute_subvolumes(img=img, mask=mask)
        if self.transform is not None:
            item = self.transform(item)

        return item

    def _compute_subvolumes(
        self, img: np.ndarray, mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        centroid = self._compute_centroid(mask=mask)
        # bounds for boolean indexing
        lower_bound, upper_bound = self._get_bounds(
            centroid=centroid, input_dims=img.shape[1:]
        )
        img = img[
            :,
            lower_bound[0] : upper_bound[0],
            lower_bound[1] : upper_bound[1],
            lower_bound[2] : upper_bound[2],
        ]
        mask = (
            mask[
                lower_bound[0] : upper_bound[0],
                lower_bound[1] : upper_bound[1],
                lower_bound[2] : upper_bound[2],
            ]
            if self.return_mask
            else None
        )
        return super(DatasetBraTSTumorCentered, self)._monaify(img=img, mask=mask)

    def _get_bounds(
        self, centroid: np.ndarray, input_dims: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        lower = (centroid - (np.array(self.sizes) / 2)).astype(int)
        upper = (centroid + (np.array(self.sizes) / 2)).astype(int)
        return np.clip(lower, 0, input_dims), np.clip(upper, 0, input_dims)

    @staticmethod
    def _compute_centroid(mask: np.ndarray) -> np.ndarray:
        return np.mean(np.argwhere(mask), axis=0).astype(int)


class MRIProcessor:
    def __init__(
        self,
        base_path: str,
        tumor_centered: bool,
        transform: "augmentations",
        modalities: List[str] = ["t1", "t1ce", "t2", "flair"],
        size: Tuple[int, ...] = (64, 64, 64),
    ) -> None:
        self.base_path = base_path
        self.size = size
        self.tumor_centered = tumor_centered
        self.modalities = modalities
        self.transform = transform

    def _compute_subvolumes(
        self, img: np.ndarray, mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        centroid = self._compute_centroid(mask=mask)
        # bounds for boolean indexing
        lower_bound, upper_bound = self._get_bounds(
            centroid=centroid, input_dims=img.shape[1:]
        )
        img = img[
            :,
            lower_bound[0] : upper_bound[0],
            lower_bound[1] : upper_bound[1],
            lower_bound[2] : upper_bound[2],
        ]
        return img, mask

    def _get_bounds(
        self, centroid: np.ndarray, input_dims: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        lower = (centroid - (np.array(self.size) / 2)).astype(int)
        upper = (centroid + (np.array(self.size) / 2)).astype(int)
        return np.clip(lower, 0, input_dims), np.clip(upper, 0, input_dims)

    @staticmethod
    def _compute_centroid(mask: np.ndarray) -> np.ndarray:
        return np.mean(np.argwhere(mask), axis=0).astype(int)

    def _load_nifti_modalities(self, base_path: str, patient: str) -> np.ndarray:
        if len(self.modalities) == 1:
            img = nib.load(
                os.path.join(base_path, patient + "-" + self.modality[0] + ".nii.gz")
            ).get_fdata()
            img = np.expand_dims(img, 0)

        else:
            early_fused = []
            for modality in self.modalities:
                early_fused.append(
                    nib.load(
                        os.path.join(base_path, patient + "-" + modality + ".nii.gz")
                    ).get_fdata()
                )

            img = np.stack(early_fused)

        return img

    def _load_mask(self, base_path: str, patient: str) -> np.ndarray:
        return nib.load(os.path.join(base_path, patient + "-seg.nii.gz")).get_fdata()

    def process(self) -> torch.Tensor:
        patient = self.base_path.split("/")[-1]
        img = self._load_nifti_modalities(base_path=self.base_path, patient=patient)
        if self.tumor_centered:
            mask = self._load_mask(base_path=self.base_path, patient=patient)
            img, _ = self._compute_subvolumes(img=img, mask=mask)

        img = self.transform(img)

        return img

class MRIEmbeddingDataset(BaseDataset):
    def __init__(self, data: pd.DataFrame, return_mask: bool = False):
        super().__init__(data, return_mask)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, bool], torch.Tensor]:
        sample = self.data.iloc[idx]
        if not pd.isna(sample["MRI"]):
            mri = torch.from_numpy(
                pd.read_csv(os.path.join(sample["MRI"], "embedding.csv")).values[0]
            ).float()
            mask = True
        else:
            mri = torch.zeros(512)
            mask = False

        if self.return_mask:
            return mri, mask
        else:
            return mri

class SurvivalMRIEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        split: pd.DataFrame,
        is_hazard_logits: bool = False
    ) -> None:
        self.dataset = MRIEmbeddingDataset(split, return_mask=False) #only MRI column with embedding paths is needed
        if is_hazard_logits:
            self.time = torch.from_numpy(split["new_time"].values)
            self.event = torch.from_numpy(split["new_event"].values)
        else:
            self.time = torch.from_numpy(split["time"].values)
            self.event = torch.from_numpy(split["event"].values)

    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        return self.dataset[idx], self.time[idx], self.event[idx]

    def __len__(self) -> int:
        return len(self.dataset)