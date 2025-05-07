import os

root_dir = "/mnt/public-datasets/drim/TCGA_all_wsi"

print("🧪 Старт dry-run проверки и удаления .svs файлов при выполнении условий\n")

# Обязательные элементы
required_elements = {
    "logs/": lambda d: os.path.isdir(os.path.join(d, "logs")),
    "patches/": lambda d: os.path.isdir(os.path.join(d, "patches")),
    "mask.npy": lambda d: os.path.isfile(os.path.join(d, "mask.npy")),
    "thumbnail.jpg": lambda d: os.path.isfile(os.path.join(d, "thumbnail.jpg")),
}

# Основной проход
for dirpath, dirnames, filenames in os.walk(root_dir):
    checks = {name: check(dirpath) for name, check in required_elements.items()}

    if all(checks.values()):
        print(f"\n✅ В папке: {dirpath}")
        print("   Проверка компонентов:")
        for name, ok in checks.items():
            full_path = os.path.join(dirpath, name)
            status = "✅ НАЙДЕНО" if ok else "❌ НЕТ"
            print(f"   - {name:<15} {status}   ({full_path})")

        # Найдём .svs файлы
        svs_files = [f for f in os.listdir(dirpath) if f.lower().endswith(".svs")]
        if svs_files:
            print("   ⚠️  Эти .svs файлы БУДУТ удалены:")
            for svs in svs_files:
                full_path = os.path.join(dirpath, svs)
                print(f"   - {full_path}")
                try:
                    os.remove(full_path)
                    print(f"     ✅ Удалено")
                except Exception as e:
                    print(f"     ❌ Ошибка при удалении: {e}")
        else:
            print("   ℹ️  .svs файлы не найдены — ничего не удаляется.")