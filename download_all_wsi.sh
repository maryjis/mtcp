#!/bin/bash

# 1. Активируем conda и нужное окружение
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

# 2. Пути
MANIFEST_FILE="/home/a.beliaeva/mtcp/wsi_id_filename.txt"
DOWNLOAD_DIR="/mnt/public-datasets/drim/TCGA_all_wsi/"

# 3. Пропустить заголовок и пройтись по каждой строке
tail -n +2 "$MANIFEST_FILE" | while IFS=$'\t' read -r uuid filename; do
    echo "⬇️ Загружаем $uuid ($filename)"
    gdc-client download "$uuid" -d "$DOWNLOAD_DIR"
done