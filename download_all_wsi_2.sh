#!/bin/bash

# 1. Активируем conda и нужное окружение
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

# 2. Пути
MANIFEST_FILE="/home/a.beliaeva/mtcp/wsi_id_filename.txt"
FILTERED_MANIFEST="/home/a.beliaeva/mtcp/wsi_id_filename.txt"
DOWNLOAD_DIR="/mnt/public-datasets/drim/TCGA_all_wsi"

# 3. Создаём новый файл с заголовком
head -n 1 "$MANIFEST_FILE" > "$FILTERED_MANIFEST"

# 4. Фильтрация: оставляем только те uuid, которых нет в DOWNLOAD_DIR
tail -n +2 "$MANIFEST_FILE" | while IFS=$'\t' read -r uuid filename; do
    if [[ ! -d "${DOWNLOAD_DIR}/${uuid}" ]]; then
        echo -e "${uuid}\t${filename}" >> "$FILTERED_MANIFEST"
    else
        echo "✅ Пропускаем $uuid ($filename): уже скачано"
    fi
done

# 5. Функция загрузки одного UUID
download_one() {
    uuid="$1"
    filename="$2"
    target_dir="${DOWNLOAD_DIR}/${uuid}"

    if [[ -d "$target_dir" ]]; then
        echo "⚠️ Повторно пропущено $uuid ($filename): директория уже есть"
    else
        echo "⬇️ Начинаем загрузку $uuid ($filename)"
        gdc-client download "$uuid" -d "$DOWNLOAD_DIR"
        echo "✅ Завершено $uuid ($filename)"
    fi
}

export -f download_one
export DOWNLOAD_DIR

# 6. Параллельная загрузка с новым манифестом
echo "🚀 Старт параллельной загрузки по отфильтрованному манифесту"
tail -n +2 "$FILTERED_MANIFEST" | \
    awk -F'\t' '{print $1 "\t" $2}' | \
    parallel --colsep '\t' -j 8 download_one {1} {2}