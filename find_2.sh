#!/bin/bash

BASE_DIR="/mnt/public-datasets/drim/TCGA_all_wsi"
# Максимальное число параллельных загрузок
MAX_JOBS=8

# Функция обработки одной UUID-папки
process_uuid_dir() {
    local uuid_dir="$1"
    local uuid=$(basename "$uuid_dir")

    # Ищем готовый .svs и незавершённый .svs.partial
    local has_svs=$(find "$uuid_dir" -maxdepth 1 -type f -name "*.svs"         | head -n 1)
    local has_partial=$(find "$uuid_dir" -maxdepth 1 -type f -name "*.svs.partial" | head -n 1)

    # Если нет .svs или есть .svs.partial — скачиваем заново
    if [[ -z "$has_svs" || -n "$has_partial" ]]; then
        echo "📂 Обрабатываю UUID-папку: $uuid_dir"
        echo "⬇️  Скачиваю https://api.gdc.cancer.gov/data/$uuid"
        (
            cd "$uuid_dir" && \
            wget --content-disposition "https://api.gdc.cancer.gov/data/$uuid"
        )
        if [[ $? -eq 0 ]]; then
            echo "✅ Успешно загружено: $uuid"
        else
            echo "❌ Ошибка загрузки: $uuid"
        fi
        echo "-----------------------------"
    fi
}

# Основной цикл по папкам первого уровня
pids=()
for uuid_dir in "$BASE_DIR"/*/; do
    process_uuid_dir "$uuid_dir" &
    pids+=("$!")
    # Ограничиваем число параллельных задач
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        sleep 1
    done
done

# Дожидаемся завершения всех фоновых задач
for pid in "${pids[@]}"; do
    wait "$pid"
done

# Удаляем все .svs.partial после загрузки
echo "🧹 Удаляю все .svs.partial..."
find "$BASE_DIR" -type f -name "*.svs.partial" -exec rm -f {} \;

echo "✅ Скрипт завершён"