import os
import numpy as np
import pyvips
from PIL import Image
import torch
import torch.nn.functional as F
import tqdm
import cv2
import argparse
import os
import tqdm
import cv2
import numpy as np
import pyvips
import json
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Tuple, Dict
import random
import requests
import io
import matplotlib.pyplot as plt
from IPython import display

def segment(img_rgba: np.ndarray, sthresh: int = 25, sthresh_up: int = 255, mthresh: int = 9, otsu: bool = True):
    """Сегментация изображения для создания маски."""
    img = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)
    if otsu:
        _, img_otsu = cv2.threshold(
            img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY
        )
    else:
        _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)
    masked_image = cv2.bitwise_and(img, img, mask=img_otsu)
    return masked_image, img_otsu


def create_thumbnail_and_mask(subdirectory_path, downscale_factor=6):
    """Создание миниатюр и масок для конкретной папки."""
    wsi_files = [f for f in os.listdir(subdirectory_path) if f.endswith("svs") or f.endswith("tif")]

    if not wsi_files:
        print(f"В папке {subdirectory_path} нет файлов .svs или .tif")
        return

    wsi_filename = wsi_files[0]
    wsi_file_path = os.path.join(subdirectory_path, wsi_filename)

    try:
        slide = pyvips.Image.new_from_file(wsi_file_path)
    except Exception as e:
        print(f"Ошибка при загрузке изображения {wsi_file_path}: {e}")
        return

    try:
        if int(float(slide.get("aperio.AppMag"))) == 40:
            d = downscale_factor + 1
        else:
            d = downscale_factor

        thumbnail = pyvips.Image.thumbnail(
            wsi_file_path,
            slide.width / (2**d),
            height=slide.height / (2**d),
        ).numpy()

        thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGBA2RGB)
        thumbnail_hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)
        mask_hsv = np.tile(thumbnail_hsv[:, :, 1] < 160, (3, 1, 1)).transpose(1, 2, 0)
        thumbnail *= mask_hsv

        masked_image, mask = segment(thumbnail)

        masked_image = Image.fromarray(masked_image).convert("RGB")
        masked_image.save(os.path.join(subdirectory_path, "thumbnail.jpg"))
        np.save(os.path.join(subdirectory_path, "mask.npy"), mask)

    except Exception as e:
        print(f"Ошибка при обработке миниатюры или маски для {subdirectory_path}: {e}")


class PatchExtractor:
    def __init__(self, num_patches, patch_size, iterations, s_min: int = 150, v_max: int = 150):
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.iterations = iterations
        self.s_min = s_min
        self.v_max = v_max

    def patch_to_score(self, patch: np.ndarray):
        mask = np.tile(patch[:, :, 1] > self.s_min, (3, 1, 1)).transpose(1, 2, 0) * np.tile(patch[:, :, 2] < self.v_max, (3, 1, 1)).transpose(1, 2, 0)
        return (mask.sum(-1) / 3).sum()

    @staticmethod
    def _from_idx_to_row_col(idx: int, width: int) -> Tuple[int, int]:
        row = (idx // width)
        col = (idx % width)
        return (row, col)

    def __call__(self, slide, mask):
        patches_buffer = {}
        factor = int(slide.width / mask.shape[1])
        delta = int(self.patch_size / factor)
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        kernel = torch.ones(1, 1, delta, delta)
        probabilities = F.conv2d(mask, kernel, stride=(delta, delta))
        probabilities /= probabilities.sum()
        probabilities = probabilities.squeeze()
        n_samples = min(torch.count_nonzero(probabilities).item(), self.iterations)
        indexes = torch.multinomial(probabilities.view(-1), n_samples, replacement=False)

        for idx in indexes:
            patch, idx = self._from_idx_to_patch(slide, idx.item(), probabilities.size(1))
            score = int(self.patch_to_score(cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)))
            patches_buffer[score] = patch

        patches_buffer = dict(sorted(patches_buffer.items(), key=lambda x: x[0], reverse=True))
        return patches_buffer

    def _from_idx_to_patch(self, slide, idx, width):
        idx = self._from_idx_to_row_col(idx, width)
        row = idx[0] * self.patch_size
        col = idx[1] * self.patch_size
        region = slide.crop(int(col), int(row), self.patch_size, self.patch_size)
        patch = np.ndarray(buffer=region.write_to_memory(),
                           dtype=np.uint8,
                           shape=(region.height, region.width, region.bands))
        return cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB), idx


def create_patches(subdirectory_path, num_patches=100, patch_size=256, iterations=1000):
    """Создание патчей для конкретной папки."""
    wsi_files = [f for f in os.listdir(subdirectory_path) if f.endswith("svs") or f.endswith("tif")]
    if not wsi_files:
        return

    wsi_filename = wsi_files[0]
    wsi_file_path = os.path.join(subdirectory_path, wsi_filename)
    mask_path = os.path.join(subdirectory_path, "mask.npy")

    if not os.path.exists(mask_path):
        print(f"Маска отсутствует для {subdirectory_path}. Пропускаем.")
        return

    try:
        slide = pyvips.Image.new_from_file(wsi_file_path)
        mask = np.load(mask_path)

        # Проверяем размеры слайда и маски
        slide_width, slide_height = slide.width, slide.height
        mask_height, mask_width = mask.shape
        if slide_width != mask_width or slide_height != mask_height:
            print(f"Несоответствие размеров маски ({mask_height}, {mask_width}) и слайда ({slide_width}, {slide_height}). Масштабируем маску...")
            mask = cv2.resize(mask, (slide_width, slide_height), interpolation=cv2.INTER_NEAREST)

        patches_folder = os.path.join(subdirectory_path, "patches")
        if os.path.exists(patches_folder):
            existing_patches = os.listdir(patches_folder)
            if len(existing_patches) != num_patches:
                print(f"Пересоздаём патчи для {subdirectory_path}.")
                for file in existing_patches:
                    os.remove(os.path.join(patches_folder, file))
                os.rmdir(patches_folder)
            else:
                print(f"Пропуск {subdirectory_path}, патчи уже существуют.")
                return

        os.makedirs(patches_folder, exist_ok=True)

        extractor = PatchExtractor(num_patches, patch_size, iterations)
        patches = extractor(slide, mask)

        for i, (score, patch) in enumerate(patches.items()):
            patch = Image.fromarray(patch)
            patch.save(os.path.join(patches_folder, f"{i}_{score}.png"))

    except Exception as e:
        print(f"Ошибка при обработке {subdirectory_path}: {e}")


if __name__ == "__main__":
    data_paths = [
        "/mnt/public-datasets/drim/wsi/8dbc7038-8501-44fd-8b66-39b50e30a178",
        "/mnt/public-datasets/drim/wsi/c0042a92-b19e-4f03-a877-ec68f6b24953"
    ]

    for path in data_paths:
        print(f"Создание миниатюр и масок для {path}...")
        create_thumbnail_and_mask(path)

        print(f"Создание патчей для {path}...")
        create_patches(path, num_patches=100, patch_size=256, iterations=1000)
