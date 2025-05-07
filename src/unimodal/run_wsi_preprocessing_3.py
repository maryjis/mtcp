import argparse
import os
import tqdm
import cv2
import numpy as np
import pyvips
import openslide
import json
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import multiprocessing as mp
import time
import logging
from typing import Tuple, Dict

# === Конфигурация логгера ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)

FAILED_WSI_LOG = "failed_wsi_files.txt"

# Для подавления сообщений libvips перенаправим stderr при вызовах
import contextlib
import sys


def safe_open_wsi(path: str) -> Tuple[str, object]:
    """Пытаемся открыть WSI через pyvips, затем openslide; иначе возвращаем None."""
    try:
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stderr(devnull):
            slide = pyvips.Image.new_from_file(path)
        logging.info(f"Открыто через pyvips: {path}")
        return 'pyvips', slide
    except pyvips.error.Error:
        logging.info(f"pyvips не смог открыть {path}")
        try:
            slide = openslide.OpenSlide(path)
            logging.info(f"Открыто через OpenSlide: {path}")
            return 'openslide', slide
        except Exception as e:
            logging.warning(f"Ошибка при создании mask для {full}: {e} — выбираем случайные патчи без маски")
            # Фоллбэк: генерируем маску из единиц (берём весь слайд)
            if engine == 'pyvips':
                mask_h = slide.height // args.patch_size
                mask_w = slide.width // args.patch_size
            else:
                dims = slide.level_dimensions[0]
                mask_h = dims[1] // args.patch_size
                mask_w = dims[0] // args.patch_size
            mask = np.ones((mask_h, mask_w), dtype=np.uint8)
            # продолжаем без return None, None


def get_masked_hsv(patch: np.ndarray) -> np.ndarray:
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    mask = (patch_hsv[:, :, 1] > 150) & (patch_hsv[:, :, 2] < 150)
    return patch * np.stack([mask]*3, axis=-1)


def segment(img_rgba: np.ndarray, sthresh: int = 25, sthresh_up: int = 255, mthresh: int = 9, otsu: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    med = cv2.medianBlur(hsv[:, :, 1], mthresh)
    if otsu:
        _, bin_mask = cv2.threshold(med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    else:
        _, bin_mask = cv2.threshold(med, sthresh, sthresh_up, cv2.THRESH_BINARY)
    masked = cv2.bitwise_and(img, img, mask=bin_mask)
    return masked, bin_mask


def sanity_check(base_path: str, num_patches: int = 100):
    for sub in tqdm.tqdm(os.listdir(base_path)):
        sp = os.path.join(base_path, sub)
        if not os.path.isdir(sp) or sub == 'logs':
            continue
        pf = os.path.join(sp, 'patches')
        if not os.path.exists(pf):
            logging.warning(f"Папка patches не найдена: {pf}")
            continue
        files = os.listdir(pf)
        if len(files) != num_patches:
            logging.warning(f"{sub}: найдено {len(files)} вместо {num_patches}")


class PatchExtractor:
    def __init__(self, num_patches: int, patch_size: int, iterations: int, s_min: int = 150, v_max: int = 150):
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.iterations = iterations
        self.s_min = s_min
        self.v_max = v_max

    def patch_to_score(self, patch: np.ndarray) -> int:
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        mask = (hsv[:, :, 1] > self.s_min) & (hsv[:, :, 2] < self.v_max)
        return int(mask.sum())

    @staticmethod
    def _from_idx_to_row_col(idx: int, width: int) -> Tuple[int, int]:
        return divmod(idx, width)

    def _from_idx_to_patch(self, engine: str, slide, idx: int, width: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        r, c = self._from_idx_to_row_col(idx, width)
        y, x = r * self.patch_size, c * self.patch_size
        if engine == 'pyvips':
            if x + self.patch_size > slide.width or y + self.patch_size > slide.height:
                raise ValueError("Patch out of bounds (pyvips)")
            region = slide.crop(x, y, self.patch_size, self.patch_size)
            arr = np.ndarray(buffer=region.write_to_memory(), dtype=np.uint8, shape=(region.height, region.width, region.bands))
            rgb = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        else:
            dims = slide.level_dimensions[0]
            if x + self.patch_size > dims[0] or y + self.patch_size > dims[1]:
                raise ValueError("Patch out of bounds (openslide)")
            patch = slide.read_region((x, y), 0, (self.patch_size, self.patch_size))
            rgb = np.array(patch)[:, :, :3]
        return rgb, (r, c)

    def __call__(self, engine: str, slide, mask: np.ndarray) -> Tuple[Dict[int, np.ndarray], None]:
        start = time.time()
        global vips_error_count
        vips_error_count = 0

        width_mask = mask.shape[1]
        if engine == 'pyvips':
            width_slide, height_slide = slide.width, slide.height
        else:
            width_slide, height_slide = slide.level_dimensions[0]
        factor = width_slide // width_mask
        delta = max(1, self.patch_size // factor)

        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        kernel = torch.ones(1, 1, delta, delta)
        probs = F.conv2d(mask_t, kernel, stride=(delta, delta)).squeeze()

        h, w = probs.shape
        idxs = [i for i in range(probs.numel()) 
                if ((i//w)*self.patch_size + self.patch_size) <= height_slide 
                and ((i%w)*self.patch_size + self.patch_size) <= width_slide]

        flat = probs.view(-1)
        valid_probs = flat[idxs].clone()
        n = min(len(idxs), self.iterations)
        if valid_probs.sum() == 0:
            chosen = np.random.choice(idxs, size=n, replace=False)
            logging.info("Маска пуста — выбираем патчи случайно")
        else:
            valid_probs /= valid_probs.sum()
            samp = torch.multinomial(valid_probs, num_samples=n, replacement=False)
            chosen = [idxs[i] for i in samp.tolist()]

        scored = []
        errors = 0
        for idx in chosen:
            try:
                patch, coord = self._from_idx_to_patch(engine, slide, idx, w)
                score = self.patch_to_score(patch)
                scored.append((score, patch))
            except Exception as e:
                errors += 1
                logging.warning(f"Ошибка патча idx={idx}: {e}")
        logging.info(f"Ошибок при извлечении патчей: {errors}/{n}")

        if not scored:
            raise RuntimeError("Не удалось извлечь ни одного патча")

        scored.sort(key=lambda x: x[0], reverse=True)
        elapsed = time.time() - start
        logging.info(f"Патчи сгенерированы за {elapsed:.2f}s; libvips errors: {vips_error_count}")
        return {s: p for s, p in scored[:self.num_patches]}, None


def process_single_case(args_tuple):
    pid, rel, args, wsi_map = args_tuple
    base = args.gbm_data_path if 'GBM' in rel else args.lgg_data_path
    full = os.path.join(base, *rel.split('/')[-2:])

    logging.info(f"--- Обработка WSI: {full} ---")
    start = time.time()

    if not os.path.exists(full):
        logging.warning(f"Файл не найден: {full}")
        return

    engine, slide = safe_open_wsi(full)
    if slide is None:
        with open(FAILED_WSI_LOG, 'a') as f:
            f.write(full + "\n")
        logging.warning(f"WSI добавлен в {FAILED_WSI_LOG}")
        return

    mask_fp = os.path.join(os.path.dirname(full), 'mask.npy')
    if not os.path.exists(mask_fp):
        try:
            logging.info(f"Генерируем thumbnail и mask для {full}")
            if engine == 'pyvips':
                thumb = slide.thumbnail_image(slide.width // (2**(args.downscale_factor + 1)))
                thumb_np = np.ndarray(buffer=thumb.write_to_memory(), dtype=np.uint8, shape=(thumb.height, thumb.width, thumb.bands))
                thumb_rgb = cv2.cvtColor(thumb_np, cv2.COLOR_RGBA2RGB)
            else:
                lvl = slide.get_best_level_for_downsample(2**args.downscale_factor)
                thumb_rgb = np.array(slide.read_region((0,0), lvl, slide.level_dimensions[lvl]))[:, :, :3]
            masked, mask = segment(thumb_rgb)
            Image.fromarray(masked).save(os.path.join(os.path.dirname(full), 'thumbnail.jpg'))
            np.save(mask_fp, mask)
            logging.info(f"Thumbnail и mask сохранены для {full}")
        except Exception as e:
            with open(FAILED_WSI_LOG, 'a') as f:
                f.write(full + "\n")
            logging.warning(f"Ошибка создания mask для {full}: {e}")
            return
    else:
        logging.info(f"Загрузка существующей mask для {full}")
        mask = np.load(mask_fp)

    extractor = PatchExtractor(args.num_patches, args.patch_size * 2, args.iterations, s_min=130, v_max=170)
    try:
        patches, _ = extractor(engine, slide, mask)
    except Exception as e:
        with open(FAILED_WSI_LOG, 'a') as f:
            f.write(full + "\n")
        logging.warning(f"Ошибка извлечения патчей для {full}: {e}")
        return

    pf = os.path.join(os.path.dirname(full), 'patches')
    os.makedirs(pf, exist_ok=True)
    saved = 0
    for i, (score, p) in enumerate(patches.items()):
        fn = os.path.join(pf, f"{i}_{score}.png")
        if not os.path.exists(fn):
            Image.fromarray(p).save(fn)
            saved += 1
    logging.info(f"Сохранено патчей: {saved}/{len(patches)} для {full}")

    sanity_check(os.path.dirname(full), num_patches=args.num_patches)
    elapsed = time.time() - start
    logging.info(f"--- Завершено {full} за {elapsed:.2f}s ---")


def main(args):
    df = pd.read_csv(args.wsi_file_path)
    with open(args.mapping_path, 'r') as f:
        wsi_map = json.load(f)

    missing_dirs = []
    for sub in os.listdir(args.gbm_data_path):
        pth = os.path.join(args.gbm_data_path, sub)
        if os.path.isdir(pth) and not os.path.isdir(os.path.join(pth, 'patches')):
            missing_dirs.append(os.path.abspath(pth))

    missing_ids = []
    for sid, rel in wsi_map.items():
        parent = os.path.abspath(os.path.dirname(rel))
        if parent in missing_dirs:
            missing_ids.append(sid)

    df = df[df['submitter_id'].isin(missing_ids)]
    logging.info(f"WSI на обработку: {len(df)}")

    id2path = {sid: wsi_map[sid] for sid in df['submitter_id'] if sid in wsi_map}
    args_list = [(pid, rel, args, wsi_map) for pid, rel in id2path.items()]

    with mp.Pool(processes=8) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(process_single_case, args_list), total=len(args_list)):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI patch extraction with INFO logging and fallback")
    parser.add_argument("--gbm_data_path", default="/mnt/public-datasets/drim/TCGA_all_wsi")
    parser.add_argument("--lgg_data_path", default="/mnt/public-datasets/drim/TCGA_all_wsi")
    parser.add_argument("--mapping_path", default="/home/a.beliaeva/mtcp/src/data/wsi_mapping.json")
    parser.add_argument("--wsi_file_path", default="/home/a.beliaeva/mtcp/dataset_with_indicators.csv")
    parser.add_argument("--num_patches", type=int, default=100)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--downscale_factor", type=int, default=6)
    args = parser.parse_args()
    main(args)
