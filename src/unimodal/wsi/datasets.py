from typing import Tuple, Union
from PIL import Image
import torch
import pandas as pd
from src.datasets import BaseDataset  # Обновлено на BaseDataset, согласно новым изменениям


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
        transform: "torchvision.transforms" = None
    ) -> None:
        super().__init__(data=dataframe, transform=transform, return_mask=return_mask)
        self.k = k
        self.is_train = is_train

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, bool], torch.Tensor]:
        sample = self.data.iloc[idx]  # используем self.data (dataframe)
        
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

        # Применяем трансформации, если они были переданы
        if self.transform:
            data = self.transform(data)

        if self.return_mask:
            return data, mask
        else:
            return data

    def __len__(self) -> int:
        return len(self.data)
