import os

# Папки для проверки
folders_to_check = [
    '/mnt/public-datasets/drim/TCGA-GBM_WSI',
    '/mnt/public-datasets/drim/wsi'
]

# Путь для сохранения результата
missing_patches_file = "/home/a.beliaeva/mtcp/src/unimodal/missing_patches.txt"
incomplete_patches_file = "/home/a.beliaeva/mtcp/src/unimodal/incomplete_patches.txt"
file_counts_file = "/home/a.beliaeva/mtcp/src/unimodal/file_counts.txt"

# Результаты
missing_patches = []
incomplete_patches = []
file_counts = []

for folder in folders_to_check:
    if os.path.isdir(folder):
        # Используем генератор для ленивой обработки подпапок
        subdirectories = (os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))
        
        for subdirectory in subdirectories:
            patches_path = os.path.join(subdirectory, 'patches')
            if not os.path.isdir(patches_path):  # Проверяем только если `patches` отсутствует
                missing_patches.append(subdirectory)
            else:
                # Проверяем количество файлов в папке `patches`
                patch_files = [f for f in os.listdir(patches_path) if os.path.isfile(os.path.join(patches_path, f))]
                file_count = len(patch_files)
                file_counts.append(f"{subdirectory}: {file_count} files")
                
                if file_count != 100:
                    incomplete_patches.append(subdirectory)
    else:
        missing_patches.append(f"Folder not found: {folder}")

# Сохранение результатов в файлы
with open(missing_patches_file, "w") as f:
    f.writelines(f"{path}\n" for path in missing_patches)

with open(incomplete_patches_file, "w") as f:
    f.writelines(f"{path}\n" for path in incomplete_patches)

with open(file_counts_file, "w") as f:
    f.writelines(f"{line}\n" for line in file_counts)

print(f"Результаты сохранены в {missing_patches_file}, {incomplete_patches_file} и {file_counts_file}")


