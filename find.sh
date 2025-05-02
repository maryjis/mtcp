#!/bin/bash

BASE_DIR="/mnt/public-datasets/drim/TCGA_all_wsi"
# Максимальное число параллельных загрузок
MAX_JOBS=4

# Функция обработки одной UUID-папки
process_uuid_dir() {
    local uuid_dir="$1"
    local uuid=$(basename "$uuid_dir")

    # Пропускаем, если есть подпапка patches
    if [ -d "$uuid_dir/patches" ]; then
        return
    fi

    # Есть ли уже .svs?
    local has_svs=$(find "$uuid_dir" -maxdepth 1 -type f -name "*.svs" | head -n 1)
    # Есть ли .svs.partial?
    local has_partial=$(find "$uuid_dir" -maxdepth 1 -type f -name "*.svs.partial" | head -n 1)

    # Запускаем, если нет .svs или есть .svs.partial
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
    # Запускаем обработку в фоне
    process_uuid_dir "$uuid_dir" &
    # Запоминаем PID фонового процесса
    pids+=("$!")
    # Если число фоновых задач >= MAX_JOBS, ждём освобождения
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        sleep 1
    done
done

# Дожидаемся завершения всех фоновых задач
for pid in "${pids[@]}"; do
    wait "$pid"
done

# После загрузки — удаляем все .svs.partial
echo "🧹 Удаляю все .svs.partial..."
find "$BASE_DIR" -type f -name "*.svs.partial" -exec rm -f {} \;

echo "✅ Скрипт завершён"