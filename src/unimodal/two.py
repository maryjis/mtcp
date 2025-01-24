import os

def check_subdirectories(base_path):
    non_svs_folders = []
    missing_svs_files = []
    
    # Перебор всех подпапок в заданной директории
    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)

        if os.path.isdir(subdir_path):  # Если это директория
            # Флаги для проверки наличия папок logs и svs
            contains_logs = False
            contains_svs = False
            
            # Перебор файлов в подпапке
            for item in os.listdir(subdir_path):
                item_path = os.path.join(subdir_path, item)
                
                if os.path.isdir(item_path) and item == 'logs':
                    contains_logs = True
                elif os.path.isdir(item_path) and item == 'svs':
                    contains_svs = True
                elif item.endswith('.svs'):
                    contains_svs = True
                else:
                    # Если файл не является .svs и не является папкой logs или svs
                    non_svs_folders.append(f"В папке {subdir_path} найден файл: {item}")

            # Проверка наличия .svs файла
            if not contains_svs:
                missing_svs_files.append(f"Нет .svs файла в папке: {subdir_path}")

    return non_svs_folders, missing_svs_files


def write_results(base_path, non_svs_folders, missing_svs_files):
    # Путь для сохранения файлов
    result_path = "/home/a.beliaeva/mtcp/src/unimodal"

    # Запись информации о папках с нежелательными файлами
    with open(os.path.join(result_path, 'non_svs_folders.txt'), 'w') as f:
        for folder in non_svs_folders:
            f.write(folder + "\n")

    # Запись информации о папках без svs файлов
    with open(os.path.join(result_path, 'missing_svs_files.txt'), 'w') as f:
        for folder in missing_svs_files:
            f.write(folder + "\n")


def main():
    # Пути к папкам
    base_paths = [
        '/mnt/public-datasets/drim/TCGA-GBM_WSI',
        '/mnt/public-datasets/drim/wsi'
    ]
    
    # Списки для записи результатов
    non_svs_folders = []
    missing_svs_files = []

    # Проверяем все указанные директории
    for base_path in base_paths:
        subdir_non_svs_folders, subdir_missing_svs_files = check_subdirectories(base_path)
        non_svs_folders.extend(subdir_non_svs_folders)
        missing_svs_files.extend(subdir_missing_svs_files)

    # Записываем результаты в текстовые файлы
    write_results(base_paths[0], non_svs_folders, missing_svs_files)


if __name__ == "__main__":
    main()
