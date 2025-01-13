from typing import Optional, Union, List, Dict, Tuple, Any
import os
import numpy as np
import nibabel as nib
from abc import abstractmethod
from torch.utils.data import Dataset
import monai
import torch
import pandas as pd
from omegaconf import OmegaConf
from src.datasets import BaseDataset
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #otherwise warnings from torch.load raise, weights_only is not an option: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly.

__all__ = ["DatasetBraTSTumorCentered"]

class DatasetBraTSTensor(Dataset):
    def __init__(
        self,
        patients: List[str],
        tag: str = "roi",
        modalities: Union[str, List[str]] = "t1c",
        size: Union[List[int], Tuple[int], Tuple[int, int], Tuple[int, int, int]] = (64, 64, 64),
    ) -> None:
        self.patients = patients
        self.modalities = modalities
        self.size = size
        self.tag = tag

    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx):
        return torch.load(
            os.path.join(
                self.patients[idx],
                f"{self.tag}_{'_'.join(self.modalities)}_{'_'.join(list(map(str, self.size)))}.pt"
            ),
            weights_only=False
        )

class _BaseDatasetBraTS(Dataset):
    def __init__(
        self,
        path: str,
        modality: Union[List[str]],
        patients: Optional[List[str]],
        return_mask: bool,
        transform: Optional["monai.transforms"] = None
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


class MRIDataset(BaseDataset):
    def __init__(self, 
                 data: pd.DataFrame,
                 modality: Union[List[str]],
                 sizes: Tuple[int, ...],
                 transform = None ,
                 is_hazard_logits = False,
                 return_mask: bool = False,
                 tensor_name: str = None):
        super().__init__(data, None, transform, is_hazard_logits, return_mask)
        self.sizes = sizes
        self.modality = modality
        self.tensor_name = tensor_name

    def _load_nifti_modalities(self, patient: str) -> np.ndarray:
        patient_tag = patient.split(os.sep)[-1]
        if len(self.modality) == 1:
            img = nib.load(
                os.path.join(
                    patient, patient_tag + "-" + self.modality[0] + ".nii.gz"
                )
            ).get_fdata()
            img = np.expand_dims(img, 0)

        else:
            early_fused = []
            for modality in self.modality:
                early_fused.append(
                    nib.load(
                        os.path.join(
                            patient, patient_tag + "-" + modality + ".nii.gz"
                        )
                    ).get_fdata()
                )

            img = np.stack(early_fused)

        return img
    
    @staticmethod
    def _monaify(img: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        item = {"image": img}
        if mask is not None:
            item["mask"] = mask

        return item

    def _load_mask(self, patient: str) -> np.ndarray:
        patient_tag = patient.split(os.sep)[-1]
        return nib.load(
            os.path.join(patient, patient_tag + "-seg.nii.gz")
        ).get_fdata()
    
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
        
        return self._monaify(img=img)

    def _get_bounds(
        self, centroid: np.ndarray, input_dims: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        lower = (centroid - (np.array(self.sizes) / 2)).astype(int)
        upper = (centroid + (np.array(self.sizes) / 2)).astype(int)
        return np.clip(lower, 0, input_dims), np.clip(upper, 0, input_dims)

    @staticmethod
    def _compute_centroid(mask: np.ndarray) -> np.ndarray:
        return np.mean(np.argwhere(mask), axis=0).astype(int)

    
    def __getitem__(self, idx: int):
        sample = self.data.iloc[idx]["MRI"]
        mask = False
        if not pd.isna(sample):
            mask = True
            item = None
            if self.tensor_name is not None:
                item = torch.load(
                    os.path.join(
                        sample,
                        f"{self.tensor_name}_{'_'.join(self.modality)}_{'_'.join(list(map(str, self.sizes)))}.pt"
                    ),
                    weights_only=False
                )
            
            else:
                mask_img = self._load_mask(patient=sample)
                img = self._load_nifti_modalities(patient=sample)
                item = self._compute_subvolumes(img=img, mask=mask_img)['image']
            
            if self.transform is not None:
                item = self.transform(item)

            if self.return_mask:
                return item, mask
            else:
                return item
            
        else:
            if self.return_mask:
                return torch.zeros(OmegaConf.to_object(self.sizes)).unsqueeze(0), mask
            else:
                return torch.zeros(OmegaConf.to_object(self.sizes)).unsqueeze(0)

class MRISurvivalDataset(MRIDataset):
    def __init__(self, data: pd.DataFrame, modality: Union[str, List[str]], sizes: Tuple[int, ...],
                 transform = None, is_hazard_logits = False, return_mask: bool = False, tensor_name: str = None):
        super().__init__(data, modality, sizes, transform, is_hazard_logits, return_mask, tensor_name)
        if is_hazard_logits:
            self.time = torch.from_numpy(data["new_time"].values)
            self.event = torch.from_numpy(data["new_event"].values)
        else:
            self.time = torch.from_numpy(data["time"].values)
            self.event = torch.from_numpy(data["event"].values)

    def __getitem__(self, idx: int):
        if self.return_mask:
            item, mask = super().__getitem__(idx)
            return item, mask, self.time[idx], self.event[idx]
        else:
            item = super().__getitem__(idx)
            return item, self.time[idx], self.event[idx]

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
                os.path.join(base_path, patient + "-" + self.modalities[0] + ".nii.gz")
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
    def __init__(
        self, 
        data: pd.DataFrame, 
        return_mask: bool = False,
        embedding_name: str = "embedding",
    ):
        super().__init__(data, return_mask)
        self.embedding_name = embedding_name

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, bool], torch.Tensor]:
        sample = self.data.iloc[idx]
        if not pd.isna(sample["MRI"]):
            mri = torch.from_numpy(
                pd.read_csv(os.path.join(sample["MRI"], f"{self.embedding_name}.csv")).values[0]
            ).float()
            mask = True
        else:
            mri = torch.zeros(512)
            mask = False

        if self.return_mask:
            return mri, mask
        else:
            return mri

class SurvivalMRIDataset(torch.utils.data.Dataset):
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
        return self.dataset[idx], self.time[idx], self.event[idx]

    def __len__(self) -> int:
        return len(self.dataset)