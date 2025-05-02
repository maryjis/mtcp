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

# ------------------ Вспомогательные функции ------------------

def get_masked_hsv(patch: np.ndarray):
    """Маскировка по HSV (не используется напрямую в примере)."""
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
        region = slide.crop(x, y, self.patch_size, self.patch_size)
        arr = np.ndarray(buffer=region.write_to_memory(),
                         dtype=np.uint8,
                         shape=(region.height, region.width, region.bands))
        rgb = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        return rgb, (r, c)

    def __call__(self, slide, mask):
        factor = slide.width // mask.shape[1]
        delta = self.patch_size // factor

        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        kernel = torch.ones(1, 1, delta, delta)
        probs = F.conv2d(mask_t, kernel, stride=(delta, delta)).squeeze()
        n = min(int((probs>0).sum()), self.iterations)
        idxs = torch.multinomial(probs.view(-1), n, replacement=False)

        scored = []
        for idx in idxs:
            patch, _ = self._from_idx_to_patch(slide, idx, probs.size(1))
            score = int(self.patch_to_score(patch))
            scored.append((score, patch))
        # Сортируем по убыванию score и берём топ-N
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:self.num_patches]
        return {s: p for s, p in top}, None

def segment(
    img_rgba: np.ndarray,
    sthresh: int = 25,
    sthresh_up: int = 255,
    mthresh: int = 9,
    otsu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    med = cv2.medianBlur(hsv[:, :, 1], mthresh)
    if otsu:
        _, bin_mask = cv2.threshold(med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    else:
        _, bin_mask = cv2.threshold(med, sthresh, sthresh_up, cv2.THRESH_BINARY)
    masked = cv2.bitwise_and(img, img, mask=bin_mask)
    return masked, bin_mask

# ------------------ Основные шаги ------------------

def create_thumbnail_and_mask(data_path, downscale_factor=6):
    """Генерация thumbnail.jpg и mask.npy, пропускаем, если они уже есть."""
    for sub in tqdm.tqdm(os.listdir(data_path)):
        subp = os.path.join(data_path, sub)
        if not os.path.isdir(subp):
            continue

        thumb_fp = os.path.join(subp, "thumbnail.jpg")
        mask_fp  = os.path.join(subp, "mask.npy")
        if os.path.exists(thumb_fp) and os.path.exists(mask_fp):
            print(f"[SKIP] {sub}: thumbnail и mask уже существуют")
            continue

        # ищем WSI-файл
        files = [f for f in os.listdir(subp) if f.endswith(("svs","tif"))]
        if not files:
            print(f"[NO WSI] {sub}")
            continue
        wsi = os.path.join(subp, files[0])
        try:
            slide = pyvips.Image.new_from_file(wsi)
        except Exception as e:
            print(f"[ERR] загрузка {wsi}: {e}")
            continue

        mag = int(float(slide.get("aperio.AppMag") or 0))
        d = downscale_factor + 1 if mag == 40 else downscale_factor
        thumb = pyvips.Image.thumbnail(wsi, slide.width//(2**d), height=slide.height//(2**d)).numpy()
        thumb = cv2.cvtColor(thumb, cv2.COLOR_RGBA2RGB)
        masked, mask = segment(thumb)
        Image.fromarray(masked).save(thumb_fp)
        np.save(mask_fp, mask)
        print(f"[OK] создано {thumb_fp}, {mask_fp}")

def sanity_check(base_path, num_patches=100):
    """Проверка числа и размера патчей."""
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
                arr = np.array(Image.open(os.path.join(pf,f)))
                if arr.shape != (256,256,3):
                    print(f"[SIZE] {f}: {arr.shape}")
            except Exception as e:
                print(f"[ERR] загрузка патча {f}: {e}")

def load_and_filter_wsi_data(mapping_file, df, gbm_path, lgg_path):
    with open(mapping_file, 'r') as f:
        m = json.load(f)
    if 'submitter_id' not in df.columns:
        raise KeyError("В CSV нет столбца 'submitter_id'")
    sub_ids = set(df['submitter_id'].astype(str))
    gm = {
        k: v for k, v in m.items()
        if k in sub_ids and os.path.exists(os.path.join(gbm_path, *v.split('/')[-2:]))
    }
    lg = {
        k: v for k, v in m.items()
        if k in sub_ids and os.path.exists(os.path.join(lgg_path, *v.split('/')[-2:]))
    }
    print(f"Файлов GBM: {len(gm)}, LGG: {len(lg)}")
    return {**gm, **lg}

def main(args):
    # 1) Чтение CSV и запоминание старого WSI
    df = pd.read_csv(args.wsi_file_path)
    orig_wsi = df['WSI'].copy() if 'WSI' in df.columns else None

    # 2) Фильтрация путей
    file_map = load_and_filter_wsi_data(
        args.mapping_path, df, args.gbm_data_path, args.lgg_data_path
    )

    # 3) Генерация миниатюр и масок (пропуск существующих)
    create_thumbnail_and_mask(args.gbm_data_path,   downscale_factor=args.downscale_factor)
    create_thumbnail_and_mask(args.lgg_data_path,   downscale_factor=args.downscale_factor)

    # 4) Построение id2path (выбор вручную при множестве вариантов)
    id2path: Dict[str,str] = {}
    for pid, path in file_map.items():
        if isinstance(path, (list, tuple)):
            # показать thumbnails и спросить индекс
            thumbs = []
            for p in path:
                thumbs.append(Image.open(os.path.join(os.path.dirname(p), 'thumbnail.jpg')))
            # тут можно вставить свой UI для выбора, а пока простой input:
            idx = int(input(f"Выберите индекс для {pid} (0..{len(path)-1}): "))
            id2path[pid] = path[idx]
        else:
            id2path[pid] = path

    # 5) Сохранение обновлённого mapping.json
    with open(args.mapping_path, 'w') as f:
        json.dump(id2path, f, indent=2)

    # 6) Обновление CSV: map + fillna, без удаления столбца
    df['WSI'] = df['submitter_id'].map(id2path)
    if orig_wsi is not None:
        df['WSI'] = df['WSI'].fillna(orig_wsi)
    df.to_csv(args.wsi_file_path, index=False)
    print(f"[CSV] Обновлён {args.wsi_file_path}")

    # 7) Извлечение патчей
    for pid, rel in tqdm.tqdm(id2path.items()):
        base = args.gbm_data_path if 'GBM' in rel else args.lgg_data_path
        full = os.path.join(base, *rel.split('/')[-2:])
        if not os.path.exists(full):
            print(f"[MISS] {full}")
            continue

        slide = pyvips.Image.new_from_file(full)
        mask = np.load(os.path.join(os.path.dirname(full), 'mask.npy'))

        mag = int(float(slide.get('aperio.AppMag') or 0))
        if mag == 40:
            extr = PatchExtractor(args.num_patches, args.patch_size*2, args.iterations, s_min=130, v_max=170)
        else:
            extr = PatchExtractor(args.num_patches, args.patch_size,   args.iterations, s_min=130, v_max=170)

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
    parser = argparse.ArgumentParser(description="WSI patch extraction and thumbnail generation")
    parser.add_argument("--gbm_data_path", "-g", default="/mnt/public-datasets/drim/TCGA_all_wsi")
    parser.add_argument("--lgg_data_path", "-l", default="/mnt/public-datasets/drim/TCGA_all_wsi")
    parser.add_argument("--mapping_path",   "-m", default="/home/a.beliaeva/mtcp/src/data/wsi_mapping.json")
    parser.add_argument("--wsi_file_path", "-w", default="/home/a.beliaeva/mtcp/dataset_with_indicators.csv")
    parser.add_argument("--num_patches",   "-n", type=int, default=100)
    parser.add_argument("--patch_size",    "-s", type=int, default=256)
    parser.add_argument("--iterations",    "-i", type=int, default=1000)
    parser.add_argument("--downscale_factor","-d", type=int, default=6)
    args = parser.parse_args()
    main(args)