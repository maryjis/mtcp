from typing import Tuple, Union
from PIL import Image
import torch
import pandas as pd
from src.datasets import BaseDataset  # Обновлено на BaseDataset, согласно новым изменениям


class PatchDataset(torch.utils.data.Dataset):
    def __init__(
        self, filepaths: Tuple[str, ...], transforms: "torchvision.transforms"
    ) -> None:
        self.filepaths = filepaths
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.filepaths[idx]
        image = Image.open(path)
        image_1, image_2 = self.transforms(image)
        return image_1, image_2

    def __len__(self) -> int:
        return len(self.filepaths)


class WSIDataset(BaseDataset):  # Используется BaseDataset вместо _BaseDataset
    def __init__(
        self,
        data: pd.DataFrame,  # Вместо dataframe, data соответствует новым изменениям
        root_dir: str,  # Добавлен root_dir для обработки путей
        k: int,
        is_train: bool = True,
        return_mask: bool = False,
    ) -> None:
        super().__init__(data, root_dir, return_mask=return_mask)  # Используется новый конструктор
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
                data = data.iloc[: self.k]

            data = torch.from_numpy(data.values).float()
            mask = True
        else:
            data = torch.zeros(self.k, 512).float()
            mask = False

        if self.return_mask:
            return data, mask
        else:
            return data
