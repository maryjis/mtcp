#!/bin/bash

#———————————————————————————————————————————
# Скрипт для массового запуска CLAM-патчинга в 8 потоков
#———————————————————————————————————————————

# 1. Активируем conda
source /root/miniconda3/etc/profile.d/conda.sh
conda activate clam_latest

# 2. Папка CLAM и переход в неё
CLAM_DIR="/home/a.beliaeva/CLAM"
cd "$CLAM_DIR" || exit 1

# 3. Базовые директории
TCGA_BASE="/mnt/public-datasets/drim/TCGA_all_wsi"
PATCH_BASE="/mnt/public-datasets/drim/TCGA_all_wsi_CLAM_patches"

# 4. Параллелизм
MAX_JOBS=8
pids=()

# 5. Функция обработки одного образца
process_sample() {
  local SAMPLE_DIR="$1"
  local SAMPLE_ID
  SAMPLE_ID=$(basename "$SAMPLE_DIR")

  local DATA_DIRECTORY="$SAMPLE_DIR"
  local RESULTS_DIRECTORY="$PATCH_BASE/$SAMPLE_ID"
  mkdir -p "$RESULTS_DIRECTORY"

  echo "=== [$SAMPLE_ID] START ==="

  python create_patches_fp.py \
      --source     "$DATA_DIRECTORY" \
      --save_dir   "$RESULTS_DIRECTORY" \
      --patch_size 512 \
      --seg \
      --patch \
      --save_patches

  if [ $? -ne 0 ]; then
    echo "!!! [$SAMPLE_ID] ERROR"
  else
    echo "+++ [$SAMPLE_ID] OK"
  fi
}

export -f process_sample
export CLAM_DIR TCGA_BASE PATCH_BASE

# 6. Запускаем в фоне
for SAMPLE_DIR in "$TCGA_BASE"/*; do
  [ -d "$SAMPLE_DIR" ] || continue

  process_sample "$SAMPLE_DIR" &
  pids+=("$!")

  # Ждём, пока число фоновых задач не опустится ниже MAX_JOBS
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
    sleep 1
  done
done

# Дожидаемся окончания всех
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "=== All done ==="