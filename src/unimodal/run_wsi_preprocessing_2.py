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
import matplotlib.pyplot as plt
from IPython import display

def get_masked_hsv(patch: np.ndarray):
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    mask = (patch_hsv[:, :, 1] > 150) & (patch_hsv[:, :, 2] < 150)
    return patch * np.stack([mask]*3, axis=-1)

class PatchExtractor:
    def __init__(self, num_patches, patch_size, iterations, s_min: int = 150, v_max: int = 150):
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.iterations = iterations
        self.s_min = s_min
        self.v_max = v_max

    def patch_to_score(self, patch: np.ndarray):
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        mask = (hsv[:, :, 1] > self.s_min) & (hsv[:, :, 2] < self.v_max)
        return mask.sum()

    @staticmethod
    def _from_idx_to_row_col(idx: int, width: int) -> Tuple[int, int]:
        return divmod(idx, width)

    def _from_idx_to_patch(self, slide, idx, width):
        r, c = self._from_idx_to_row_col(int(idx), width)
        y, x = r * self.patch_size, c * self.patch_size

        if x + self.patch_size > slide.width or y + self.patch_size > slide.height:
            raise ValueError(f"Invalid crop at ({x}, {y}) with patch_size={self.patch_size} — out of bounds for image size {slide.width}x{slide.height}")

        region = slide.crop(x, y, self.patch_size, self.patch_size)

        if isinstance(region, np.ndarray):
            arr = region
        else:
            arr = np.ndarray(
                buffer=region.write_to_memory(),
                dtype=np.uint8,
                shape=(region.height, region.width, region.bands)
            )

        rgb = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        return rgb, (r, c)

    def __call__(self, slide, mask):
        factor = slide.width // mask.shape[1]
        delta = max(1, self.patch_size // factor)

        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        kernel = torch.ones(1, 1, delta, delta)
        probs = F.conv2d(mask_t, kernel, stride=(delta, delta)).squeeze()

        height, width = probs.shape
        valid_idxs = []
        for idx in range(probs.numel()):
            r, c = self._from_idx_to_row_col(idx, width)
            y, x = r * self.patch_size, c * self.patch_size
            if x + self.patch_size <= slide.width and y + self.patch_size <= slide.height:
                valid_idxs.append(idx)

        if len(valid_idxs) == 0:
            raise RuntimeError("Нет допустимых индексов патчей (всё изображение вне границ или маска пуста).")

        probs_flat = probs.view(-1)
        valid_probs = probs_flat[valid_idxs].clone().float()

        n = min(len(valid_idxs), self.iterations)

        if valid_probs.sum() <= 0:
            print(f"[FALLBACK] Маска слишком разреженная или пустая — переходим к случайному выбору без маски.")
            final_idxs = np.random.choice(valid_idxs, size=n, replace=False)
        else:
            valid_probs /= valid_probs.sum()
            sampled_idxs = torch.multinomial(valid_probs, n, replacement=False)
            final_idxs = [valid_idxs[i.item()] for i in sampled_idxs]

        scored = []
        for idx in final_idxs:
            try:
                patch, _ = self._from_idx_to_patch(slide, idx, width)
                score = int(self.patch_to_score(patch))
                scored.append((score, patch))
            except Exception as e:
                print(f"[SKIP] Ошибка извлечения патча по индексу {idx}: {e}")

        if len(scored) == 0:
            raise RuntimeError("Не удалось получить ни одного допустимого патча после выборки.")

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:self.num_patches]
        return {s: p for s, p in top}, None

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

def sanity_check(base_path, num_patches=100):
    for sub in tqdm.tqdm(os.listdir(base_path)):
        sp = os.path.join(base_path, sub)
        if not os.path.isdir(sp) or sub == 'logs':
            continue
        pf = os.path.join(sp, 'patches')
        if not os.path.exists(pf):
            print(f"[MISS] папка патчей не найдена: {pf}")
            continue
        files = os.listdir(pf)
        if len(files) != num_patches:
            print(f"[WARN] для {sub}: ожидалось {num_patches}, есть {len(files)}")
        for f in files:
            try:
                arr = np.array(Image.open(os.path.join(pf, f)))
                if arr.shape != (256, 256, 3):
                    print(f"[SIZE] {f}: {arr.shape}")
            except Exception as e:
                print(f"[ERR] загрузка патча {f}: {e}")

def main(args):
    df = pd.read_csv(args.wsi_file_path)

    with open(args.mapping_path, 'r') as f:
        wsi_map = json.load(f)

    # --- 1. Найдём ID без 'patches/' ---
    base_path = args.gbm_data_path
    missing_dirs = []
    for subfolder in os.listdir(base_path):
        subfolder_path = os.path.join(base_path, subfolder)
        if os.path.isdir(subfolder_path):
            patches_path = os.path.join(subfolder_path, 'patches')
            if not os.path.isdir(patches_path):
                missing_dirs.append(os.path.abspath(subfolder_path))

    missing_ids = []
    for sid, path in wsi_map.items():
        parent_dir = os.path.abspath(os.path.dirname(path))
        if parent_dir in set(missing_dirs):
            missing_ids.append(sid)

    df = df[df['submitter_id'].isin(missing_ids)]
    print(f"[AUTO] Обрабатываются только отсутствующие: {len(df)} WSI")

    with open("missing_submitter_ids.json", 'w') as f:
        json.dump(missing_ids, f, indent=2)

    id2path: Dict[str, str] = {sid: wsi_map[sid] for sid in df['submitter_id'] if sid in wsi_map}

    for pid, rel in tqdm.tqdm(id2path.items()):
        base = args.gbm_data_path if 'GBM' in rel else args.lgg_data_path
        full = os.path.join(base, *rel.split('/')[-2:])
        if not os.path.exists(full):
            print(f"[MISS] {full}")
            continue

        try:
            slide = pyvips.Image.new_from_file(full)
        except pyvips.error.Error as e:
            print(f"[SKIP] pyvips could not open file: {full}\n  ↳ Reason: {e}")
            continue
        mask_fp = os.path.join(os.path.dirname(full), 'mask.npy')
        if not os.path.exists(mask_fp):
            thumb = pyvips.Image.thumbnail(full, slide.width // (2**(args.downscale_factor + 1)))
            thumb = cv2.cvtColor(thumb.numpy(), cv2.COLOR_RGBA2RGB)
            masked, mask = segment(thumb)
            Image.fromarray(masked).save(os.path.join(os.path.dirname(full), 'thumbnail.jpg'))
            np.save(mask_fp, mask)
        else:
            mask = np.load(mask_fp)

        try:
            mag = int(float(slide.get('aperio.AppMag')))
        except pyvips.error.Error:
            print(f"[WARN] No aperio.AppMag found for {os.path.basename(slide.filename)} — defaulting to mag=40")
            mag = 40       
        extr = PatchExtractor(
            args.num_patches,
            args.patch_size * 2 if mag == 40 else args.patch_size,
            args.iterations,
            s_min=130,
            v_max=170
        )

        patches, _ = extr(slide, mask)
        patches = dict(sorted(patches.items(), key=lambda x: x[0], reverse=True))
        sel = dict(list(patches.items())[:args.num_patches])

        pf = os.path.join(os.path.dirname(full), 'patches')
        os.makedirs(pf, exist_ok=True)
        for i, (score, p) in enumerate(sel.items()):
            fn = os.path.join(pf, f"{i}_{score}.png")
            if os.path.exists(fn):
                continue
            Image.fromarray(p).save(fn)

        sanity_check(os.path.dirname(full), num_patches=args.num_patches)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI patch extraction for missing cases")
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