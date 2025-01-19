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

# Вспомогательные функции для обработки WSI и извлечения патчей

def request_file_info(data_type, base_path):
    """Запрос информации о файлах WSI из GDC"""
    fields = [
        'file_name',
        'cases.submitter_id',
        'cases.samples.sample_type',
        'cases.project.project_id',
        'cases.project.primary_site',
    ]
    fields = ','.join(fields)
    files_endpt = 'https://api.gdc.cancer.gov/files'
    filters = {
        'op': 'and',
        'content': [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.primary_site",
                    "value": ["Brain"]
                }
            },
            {
                'op': 'in',
                'content': {
                    'field': 'files.experimental_strategy',
                    'value': [data_type]
                }
            }
        ]
    }
    params = {
        'filters': filters,
        'fields': fields,
        'format': 'TSV',
        'size': '200000'
    }
    response = requests.post(files_endpt, headers={'Content-Type': 'application/json'}, json=params)
    return pd.read_csv(io.StringIO(response.content.decode('utf-8')), sep=',')


def make_patient_file_map(df, base_dir):
    """Создание маппинга файлов для пациентов"""
    d = {}
    for _, row in df.iterrows():
        patient = row['cases.0.submitter_id']
        file_path = os.path.join(base_dir, row['id'], row['file_name'])
        if os.path.exists(file_path):
            if patient in d:
                if not isinstance(d[patient], tuple):
                    d[patient] = (d[patient], file_path)
                else:
                    d[patient] += (file_path,)
            else:
                d[patient] = file_path
    return d


def get_masked_hsv(patch: np.ndarray):
    """Функция для маскировки с использованием HSV"""
    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    mask = np.tile(patch[:, :, 1] > 150, (3, 1, 1)).transpose(1, 2, 0) * np.tile(patch[:, :, 2] < 150, (3, 1, 1)).transpose(1, 2, 0)
    return patch * mask


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
        idx_buffer = {}
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
            idx_buffer[score] = idx
            idx_buffer = dict(sorted(idx_buffer.items(), key=lambda x: x[0], reverse=True))
        return patches_buffer, idx_buffer

    def _from_idx_to_patch(self, slide, idx, width):
        idx = self._from_idx_to_row_col(idx, width)
        row = idx[0] * self.patch_size
        col = idx[1] * self.patch_size
        region = slide.crop(int(col), int(row), self.patch_size, self.patch_size)
        patch = np.ndarray(buffer=region.write_to_memory(),
                           dtype=np.uint8,
                           shape=(region.height, region.width, region.bands))
        return cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB), idx


# Функция для создания миниатюр и масок

def segment(
    img_rgba: np.ndarray,
    sthresh: int = 25,
    sthresh_up: int = 255,
    mthresh: int = 9,
    otsu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)  # Apply median blurring
    if otsu:
        _, img_otsu = cv2.threshold(
            img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY
        )
    else:
        _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)
    masked_image = cv2.bitwise_and(img, img, mask=img_otsu)
    return masked_image, img_otsu


def create_thumbnail_and_mask(data_path, downscale_factor=6):
    """Создание миниатюр и масок"""
    subdirectories = os.listdir(data_path)
    for subdirectory in tqdm.tqdm(subdirectories):
        subdirectory_path = os.path.join(data_path, subdirectory)
        filenames = os.listdir(subdirectory_path)
        wsi_filename = [f for f in filenames if f.endswith("svs") or f.endswith("tif")][0]
        slide = pyvips.Image.new_from_file(os.path.join(subdirectory_path, wsi_filename))
        if int(float(slide.get("aperio.AppMag"))) == 40:
            d = downscale_factor + 1
        else:
            d = downscale_factor
        thumbnail = pyvips.Image.thumbnail(
            os.path.join(subdirectory_path, wsi_filename),
            slide.width / (2**d),
            height=slide.height / (2**d),
        ).numpy()

        thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGBA2RGB)
        thumbnail_hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)
        # to filter out felt-tip marks
        mask_hsv = np.tile(thumbnail_hsv[:, :, 1] < 160, (3, 1, 1)).transpose(1, 2, 0)
        thumbnail *= mask_hsv
        masked_image, mask = segment(thumbnail)
        masked_image = Image.fromarray(masked_image).convert("RGB")
        # save
        masked_image.save(os.path.join(subdirectory_path, "thumbnail.jpg"))
        np.save(os.path.join(subdirectory_path, "mask.npy"), mask)



def sanity_check(base_path, num_patches=100):
    """Проверка на корректность извлеченных патчей"""
    for subdirectory in tqdm.tqdm(os.listdir(base_path)):
        subdirectory_path = os.path.join(base_path, subdirectory)
        
        # Пропускаем, если это не директория или если это папка с именем 'logs'
        if not os.path.isdir(subdirectory_path) or subdirectory == 'logs':
            continue
        
        # Далее обработка только тех, что являются директориями и не имеют имя 'logs'
        patches_folder = os.path.join(subdirectory_path, 'patches')
        
        # Проверяем наличие папки с патчами
        if os.path.exists(patches_folder):
            if len(os.listdir(patches_folder)) != num_patches:
                print(f"Warning: Abnormal number of patches for {subdirectory}. Expected {num_patches}, found {len(os.listdir(patches_folder))}.")
            
            # Перебираем все файлы в папке с патчами
            for patch_file in os.listdir(patches_folder):
                patch_path = os.path.join(patches_folder, patch_file)
                
                # Загружаем каждый патч
                try:
                    patch = np.array(Image.open(patch_path))
                    # Проверяем размер патча
                    if patch.shape != (256, 256, 3):
                        print(f"Abnormal patch size for {patch_file}. Expected (256, 256, 3), got {patch.shape}.")
                except Exception as e:
                    print(f"Ошибка при загрузке патча: {patch_path}. Ошибка: {str(e)}")
        else:
            print(f"Папка с патчами не найдена: {patches_folder}")




def load_and_filter_wsi_data(mapping_file, dataframe, gbm_data_path, lgg_data_path):
    """Загрузка и фильтрация данных WSI"""
    # Загрузим ваш JSON файл
    with open(mapping_file, 'r') as f:
        wsi_mapping = json.load(f)

    print("Столбцы в dataframe:", dataframe.columns)
    print("Первые строки данных:", dataframe.head())

    # Если submitter_id существует в dataframe
    if 'submitter_id' in dataframe.columns:
        print(f"Unique submitter_ids in dataframe: {dataframe['submitter_id'].nunique()}")
    else:
        print("Ошибка: столбец 'submitter_id' не найден в dataframe.")
    
    # Фильтруем файлы из JSON по submitter_id
    # Используем правильный путь для GBM и LGG
    file_map_gbm = {
        k: v for k, v in wsi_mapping.items() 
        if k in dataframe['submitter_id'].values and os.path.exists(os.path.join(gbm_data_path, v.split('/')[-2], v.split('/')[-1]))
    }
    file_map_lgg = {
        k: v for k, v in wsi_mapping.items() 
        if k in dataframe['submitter_id'].values and os.path.exists(os.path.join(lgg_data_path, v.split('/')[-2], v.split('/')[-1]))
    }

    print(f"Количество файлов для GBM: {len(file_map_gbm)}")
    print(f"Количество файлов для LGG: {len(file_map_lgg)}")

    return {**file_map_gbm, **file_map_lgg}

def main(args):
    # Загружаем маппинг WSI
    dataframe = pd.read_csv(args.wsi_file_path, sep=',')
    
    # Передаем правильные пути для обоих наборов данных (GBM и LGG)
    file_map = load_and_filter_wsi_data(
        args.mapping_path, dataframe, args.gbm_data_path, args.lgg_data_path
    )
    print("Проверка путей:", file_map)
    
    # Создаем миниатюры и маски для каждого пациента
    create_thumbnail_and_mask(args.gbm_data_path, downscale_factor=args.downscale_factor)  # Заменили base_path на gbm_data_path
    create_thumbnail_and_mask(args.lgg_data_path, downscale_factor=args.downscale_factor)  # Для LGG
    
    # Создаем и сохраняем словарь id2path
    id2path = {}
    for patient_id, path in file_map.items():
        if isinstance(path, tuple):
            # Если несколько слайдов, отображаем их с миниатюрами
            n_slides = len(path)
            plt.figure(figsize=(12, 12))
            for i in range(n_slides):
                plt.subplot(1, n_slides, i+1)
                plt.axis('off')
                plt.title(pyvips.Image.new_from_file(path[i]).get('aperio.AppMag'))
                thumbnail_path = '/'.join(path[i].split('/')[:-1])
                thumbnail = np.array(Image.open(os.path.join(thumbnail_path, 'thumbnail.jpg')))
                plt.imshow(thumbnail)
            display.display(plt.gcf())
            display.clear_output(wait=True)
            plt.show()
            idx = int(input())
            id2path[patient_id] = path[idx]
        else:
            id2path[patient_id] = path

    # Перезаписываем файл wsi_mapping.json
    with open(args.mapping_path, 'w') as f:
        json.dump(id2path, f)

    # Считываем данные JSON и объединяем их с dataframe
    with open(args.mapping_path, 'r') as f:
        wsi_mapping = json.load(f)

    WSI_mapping = pd.DataFrame([(k, v) for k, v in wsi_mapping.items()], columns=('submitter_id', 'WSI'))
    print(WSI_mapping)
    dataframe = dataframe.merge(WSI_mapping, how='left', on='submitter_id')
    dataframe.to_csv(args.wsi_file_path, index=False)

    # Обрабатываем данные GBM и LGG
    num_patches = args.num_patches
    patch_size = args.patch_size
    iterations = args.iterations

    # Обрабатываем каждый слайд пациента
    for patient, wsi_path in tqdm.tqdm(file_map.items()):
        # Проверяем, из какого набора данных файл
        if 'GBM' in wsi_path:
            base_path = args.gbm_data_path
        else:
            base_path = args.lgg_data_path
        
        # Путь к файлу .svs
        wsi_full_path = os.path.join(base_path, *wsi_path.split('/')[1:])
        
        # Проверяем, существует ли файл .svs
        if not os.path.exists(wsi_full_path):
            print(f"Ошибка: Файл {wsi_full_path} не существует.")
            continue
        
        print(f"Проверка пути: {wsi_full_path}")
        
        # Открываем слайд
        try:
            slide = pyvips.Image.new_from_file(wsi_full_path)  # Открываем файл слайд
            print(f"Изображение загружено успешно: {wsi_full_path}")
        except Exception as e:
            print(f"Ошибка при загрузке изображения: {wsi_full_path} \nОшибка: {str(e)}")
            continue

        # Путь к папке для загрузки маски
        folder_path = os.path.dirname(wsi_full_path)  # Путь к папке с файлом .svs
        mask_path = os.path.join(folder_path, 'mask.npy')
        
        # Загружаем маску
        try:
            mask = np.load(mask_path)
            print(f"Маска загружена: {mask_path}")
        except Exception as e:
            print(f"Ошибка при загрузке маски: {mask_path} \nОшибка: {str(e)}")
            continue
        
        # Применяем правильный extractor в зависимости от magnification
        if int(float(slide.get('aperio.AppMag'))) == 40:
            extractor = PatchExtractor(num_patches=num_patches, patch_size=patch_size*2, iterations=iterations, s_min=130, v_max=170)
        else:
            extractor = PatchExtractor(num_patches=num_patches, patch_size=patch_size, iterations=iterations, s_min=130, v_max=170)
        
        # Извлекаем патчи
        patches, _ = extractor(slide, mask)

        # Если 'patches' - это кортеж, а не словарь, обработаем его как нужно
        if isinstance(patches, dict):
            patches = dict(sorted(patches.items(), key=lambda x: x[0], reverse=True))
        else:
            print("Warning: patches не является словарем, проверим структуру данных.")
            print(f"Тип patches: {type(patches)}")
            # Если patches - это кортеж, то можно сделать что-то с первым элементом:
            patches = patches[0]  # Если patches - это кортеж, предположим, что патчи в первом элементе
            patches = dict(sorted(patches.items(), key=lambda x: x[0], reverse=True))

        selected_patches = {score: patch for score, patch in list(patches.items())[:num_patches]}
        
        # Создаем папку для патчей, если её нет
        patches_folder = os.path.join(folder_path, 'patches')
        print(f"folder_path: {folder_path}")
        print(f"patches_folder: {patches_folder}")

        if not os.path.exists(patches_folder):
            os.makedirs(patches_folder)
        
        # Сохраняем патчи
        for i, (score, patch) in enumerate(selected_patches.items()):
            patch = Image.fromarray(patch)
            patch.save(os.path.join(patches_folder, f'{int(i)}_{int(score)}.png'))
        
        # Проверка корректности
        sanity_check(folder_path)  # Передаем правильный путь




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI patch extraction and thumbnail generation")
    parser.add_argument("--base_path", "-b", default="/home/belyaeva.a/WSI_GBMLGG", help="Base path for processed data (thumbnails, masks, etc.)")
    parser.add_argument("--gbm_data_path", "-g", default="/home/belyaeva.a/TCGA-GBM_WSI", help="Path to the GBM data folder")
    parser.add_argument("--lgg_data_path", "-l", default="/home/belyaeva.a/wsi", help="Path to the LGG data folder")
    parser.add_argument("--mapping_path", "-m", default="/home/belyaeva.a/mtcp/src/data/wsi_mapping.json", help="Path to WSI mapping file")
    parser.add_argument("--num_patches", "-n", type=int, default=5, help="Number of patches to extract (1000)")
    parser.add_argument("--patch_size", "-s", type=int, default=256, help="Size of the patches (256x256)")
    parser.add_argument("--iterations", "-i", type=int, default=6, help="Number of iterations for patch extraction")
    parser.add_argument("--wsi_file_path", "-w", default="/home/belyaeva.a/mtcp/src/data/dataset.csv", help="Path to the WSI files metadata")
    parser.add_argument("--downscale_factor", "-d", type=int, default=6, help="Downscale factor for thumbnail generation")

    args = parser.parse_args()
    main(args)
