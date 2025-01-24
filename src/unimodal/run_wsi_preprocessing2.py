import os
import tqdm
import numpy as np
import pyvips
from PIL import Image
from typing import Tuple
import torch
import torch.nn.functional as F
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

# Класс для извлечения патчей
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
        probabilities = probabilities.squeeze()
        n_samples = torch.argwhere(probabilities).size(0) if torch.argwhere(probabilities).size(0) < self.iterations else self.iterations
        indexes = torch.multinomial(probabilities.view(-1), n_samples, replacement=False)
        for idx in indexes:
            patch, idx = self._from_idx_to_patch(slide, idx, probabilities.size(1))
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


def sanity_check(folder_path, num_patches):
    """Проверка корректности патчей и пересоздание при необходимости"""
    patches_folder = os.path.join(folder_path, 'patches')

    if os.path.exists(patches_folder):
        existing_patches = os.listdir(patches_folder)
        if len(existing_patches) != num_patches:
            print(f"Папка {patches_folder} содержит неправильное количество патчей ({len(existing_patches)}). Пересоздаём...")
            # Удаляем папку
            for file in existing_patches:
                os.remove(os.path.join(patches_folder, file))
            os.rmdir(patches_folder)
            os.makedirs(patches_folder)
            return False  # Нужно пересоздать
        else:
            print(f"Папка {patches_folder} содержит правильное количество патчей. Пропускаем.")
            return True  # Пропускаем

    # Если папки нет, её нужно создать
    return False


def extract_patches(data_paths, num_patches, patch_size, iterations):
    """Основной цикл извлечения патчей из нескольких папок"""
    for data_path in data_paths:
        subdirectories = os.listdir(data_path)

        for subdirectory in tqdm.tqdm(subdirectories):
            subdirectory_path = os.path.join(data_path, subdirectory)

            if not os.path.isdir(subdirectory_path):
                continue  # Пропускаем не директории

            wsi_files = [f for f in os.listdir(subdirectory_path) if f.endswith("svs") or f.endswith("tif")]
            if not wsi_files:
                continue  # Пропускаем, если нет подходящих файлов

            wsi_file = os.path.join(subdirectory_path, wsi_files[0])
            mask_path = os.path.join(subdirectory_path, 'mask.npy')

            if not os.path.exists(mask_path):
                print(f"Маска отсутствует для {subdirectory_path}. Пропускаем.")
                continue

            try:
                # Загружаем слайд и маску
                slide = pyvips.Image.new_from_file(wsi_file)
                mask = np.load(mask_path)

                # Проверяем папку `patches`
                if sanity_check(subdirectory_path, num_patches):
                    continue  # Если всё в порядке, пропускаем

                # Извлекаем патчи
                extractor = PatchExtractor(num_patches, patch_size, iterations)
                patches = extractor(slide, mask)

                # Сохраняем патчи
                patches_folder = os.path.join(subdirectory_path, 'patches')
                os.makedirs(patches_folder, exist_ok=True)
                for i, (score, patch) in enumerate(patches.items()):
                    patch = Image.fromarray(patch)
                    patch.save(os.path.join(patches_folder, f'{i}_{score}.png'))

            except Exception as e:
                print(f"Ошибка при обработке {subdirectory_path}: {e}")


if __name__ == "__main__":
    data_paths = [
        "/mnt/public-datasets/drim/TCGA-GBM_WSI",
        "/mnt/public-datasets/drim/wsi"
    ]
    num_patches = 100
    patch_size = 256
    iterations = 1000

    extract_patches(data_paths, num_patches, patch_size, iterations)