#!/bin/bash

# Список родительских PID, которые держат зомби
PPIDS=(1379807 1383891 1390026)

for ppid in "${PPIDS[@]}"; do
    if ps -p "$ppid" > /dev/null; then
        echo "⚠️ Убиваем родителя $ppid..."
        kill -TERM "$ppid"
        sleep 2
        if ps -p "$ppid" > /dev/null; then
            echo "❗ TERM не сработал, посылаем SIGKILL"
            kill -9 "$ppid"
        else
            echo "✅ Родитель $ppid завершился."
        fi
    else
        echo "ℹ️ Родитель $ppid уже мёртв."
    fi
done