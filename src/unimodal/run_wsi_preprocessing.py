import os
import json
import torch
import tqdm
import pyvips
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from typing import Tuple

# Параметры командной строки
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--skip_existing_patch", type=bool, default=False)
parser.add_argument("--data_path", type=str, default="/data/WSI/")
parser.add_argument("--num_patches", type=int, default=100)
parser.add_argument("--patch_size", type=int, default=256)
args = parser.parse_args()

# Загрузка данных WSI
def load_wsi_files():
    wsi_files = request_file_info(data_type='Diagnostic Slide')
    wsi_files = wsi_files[wsi_files['cases.0.project.project_id'].str.startswith('TCGA')]
    wsi_files = wsi_files[wsi_files['file_name'].str.endswith('.svs')]
    wsi_files = wsi_files[wsi_files['cases.0.samples.0.sample_type'] == 'Primary Tumor']
    return wsi_files

# Создание карты файлов
def create_file_map(wsi_files):
    file_map_gbm = make_patient_file_map(wsi_files, '/home/belyaeva.a/WSI_GBM/')
    file_map_lgg = make_patient_file_map(wsi_files, '/home/belyaeva.a/WSI/')
    return {**file_map_gbm, **file_map_lgg}

# Сегментация изображения для выделения маски
def segment(
    img_rgba: np.ndarray,
    sthresh: int = 25,
    sthresh_up: int = 255,
    mthresh: int = 9,
    otsu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
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


def create_thumbnails_and_masks(file_map, downscale_factor=6):
    """
    Создаёт миниатюры и маски для изображений WSI.
    """
    for patient_id, paths in tqdm.tqdm(file_map.items()):
        for path in paths:
            slide = pyvips.Image.new_from_file(path)

            # Обработка увеличения
            if int(float(slide.get("aperio.AppMag"))) == 40:
                d = downscale_factor + 1
            else:
                d = downscale_factor

            # Создание миниатюры
            thumbnail = pyvips.Image.thumbnail(
                path,
                slide.width / (2 ** d),
                height=slide.height / (2 ** d),
            ).numpy()

            thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGBA2RGB)
            thumbnail_hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)

            # Маскирование для удаления посторонних объектов
            mask_hsv = np.tile(thumbnail_hsv[:, :, 1] < 160, (3, 1, 1)).transpose(1, 2, 0)
            thumbnail *= mask_hsv

            # Сегментация
            masked_image, mask = segment(thumbnail)

            # Сохранение миниатюр и масок
            output_dir = os.path.dirname(path)
            masked_image = Image.fromarray(masked_image).convert("RGB")
            masked_image.save(os.path.join(output_dir, "thumbnail.jpg"))
            np.save(os.path.join(output_dir, "mask.npy"), mask)


# Класс для извлечения патчей
class PatchExtractor:
    def __init__(self, num_patches, patch_size, iterations, s_min=130, v_max=170):
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.iterations = iterations
        self.s_min = s_min
        self.v_max = v_max

    def patch_to_score(self, patch):
        mask = np.tile(patch[:, :, 1] > self.s_min, (3, 1, 1)).transpose(1, 2, 0) * np.tile(patch[:, :, 2] < self.v_max, (3, 1, 1)).transpose(1, 2, 0)
        return (mask.sum(-1) / 3).sum()

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
            row, col = divmod(idx, slide.width // self.patch_size)
            region = slide.crop(col * self.patch_size, row * self.patch_size, self.patch_size, self.patch_size)
            patch = np.ndarray(buffer=region.write_to_memory(), dtype=np.uint8, shape=(region.height, region.width, region.bands))
            score = self.patch_to_score(cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB))
            patches_buffer[score] = patch
            patches_buffer = dict(sorted(patches_buffer.items(), key=lambda x: x[0], reverse=True))
            patches_buffer = dict(list(patches_buffer.items())[:self.num_patches])

        return patches_buffer


# Основной процесс обработки WSI
def process_wsi_files():
    wsi_files = load_wsi_files()
    file_map = create_file_map(wsi_files)
    create_thumbnails_and_masks(file_map)

    with open('src/data/wsi_mapping.json', 'r') as f:
        mapping = json.load(f)

    for patient, wsi_path in tqdm.tqdm(mapping.items()):
        folder = os.path.dirname(wsi_path)

        if os.path.exists(os.path.join(folder, 'patches')) and len(os.listdir(os.path.join(folder, 'patches'))) == args.num_patches:
            logger.info(f"Done for patient: {patient}")
            continue

        try:
            slide = pyvips.Image.new_from_file(wsi_path)
            mask = np.load(os.path.join(folder, 'mask.npy'))  # Загружаем маску

            extractor = PatchExtractor(num_patches=args.num_patches, patch_size=args.patch_size, iterations=1000)
            patches = extractor(slide, mask)

            # Сохраняем патчи
            patch_folder = os.path.join(folder, 'patches')
            if not os.path.exists(patch_folder):
                os.makedirs(patch_folder)

            for i, (score, patch) in enumerate(patches.items()):
                patch_image = Image.fromarray(patch)
                patch_image.save(os.path.join(patch_folder, f'{i}_{score}.png'))

        except Exception as e:
            logger.error(f"Error processing {wsi_path}: {e}")


# Проверка завершенности обработки
def sanity_check():
    with open('src/data/wsi_mapping.json', 'r') as f:
        mapping = json.load(f)

    for patient, wsi_path in tqdm.tqdm(mapping.items()):
        folder = os.path.dirname(wsi_path)
        patches = os.listdir(os.path.join(folder, 'patches'))

        if len(patches) != args.num_patches:
            print(f'Abnormal number of patches for {patient}: {len(patches)}')

        for patch in patches:
            patch_path = os.path.join(folder, 'patches', patch)
            patch = np.array(Image.open(patch_path))
            if patch.shape != (args.patch_size, args.patch_size, 3):
                print(f'Abnormal patch size for {patch_path}, expected ({args.patch_size}, {args.patch_size}, 3), got {patch.shape}')


if __name__ == "__main__":
    process_wsi_files()
    sanity_check()