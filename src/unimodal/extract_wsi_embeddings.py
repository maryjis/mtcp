import argparse
import os
import json
import torch
import pandas as pd
import torchvision
from torchvision.transforms import ToTensor
from PIL import Image
import tqdm
import numpy as np

# Функция для очистки состояния словаря
def clean_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'module.encoder' in key:
            new_state_dict[key[15:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

# Параметры командной строки через argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Extract embeddings from WSI patches.")
    
    parser.add_argument('--model_path', type=str, default="/home/belyaeva.a/mtcp/src/data/models/wsi_encoder.pth", help="Path to the pretrained model weights.")
    parser.add_argument('--mapping_path', type=str, default="/home/belyaeva.a/mtcp/src/data/wsi_mapping.json", help="Path to the JSON mapping file for WSI images.")
    parser.add_argument('--output_dir', type=str, default="outputs/embeddings_wsi", help="Directory to save extracted embeddings.")
    parser.add_argument('--dataset1_path', type=str, default="/home/belyaeva.a/TCGA-GBM_WSI", help="Path to the first dataset directory.")
    parser.add_argument('--dataset2_path', type=str, default="/home/belyaeva.a/wsi", help="Path to the second dataset directory.")
    parser.add_argument("--data_path", type=str, default="/home/belyaeva.a/mtcp/src/data/dataset.csv", help="Path to the input dataframe.")
    parser.add_argument('--cuda', action='store_true', help="Use CUDA if available.")

    return parser.parse_args()

# Основная логика
def main():
    args = parse_args()

    # Проверяем наличие CUDA
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # Загружаем модель ResNet34 и модифицируем её
    print(f"Loading model from {args.model_path}")
    extractor = torchvision.models.resnet34(pretrained=False)
    extractor.fc = torch.nn.Identity()  # Заменяем последний слой
    state_dict = clean_state_dict(torch.load(args.model_path))  # Загружаем сохранённое состояние модели
    extractor.load_state_dict(state_dict, strict=False)
    extractor.to(device)
    extractor.eval()  # Переводим модель в режим оценки

    # Читаем маппинг WSI
    print(f"Loading mapping from {args.mapping_path}")
    with open(args.mapping_path, 'r') as f:
        mapping = json.load(f)
    
    # Проверим, что маппинг не пуст
    print(f"Loaded mapping with {len(mapping)} patients.")
    
    # Определяем путь для сохранения эмбеддингов
    output_path = os.path.join(args.output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Создаём и сохраняем эмбеддинги для обоих датасетов
    file_map = {}
    print("Processing patients...")
    for patient, path in tqdm.tqdm(mapping.items(), total=len(mapping)):
        print(f"Processing patient: {patient}")

        # Путь к патчам из маппинга, добавляем корневой путь для датасета
        if "TCGA-GBM_WSI" in path:
            patches_path = os.path.join(args.dataset1_path, path)
        elif "wsi" in path:  # Для второго датасета
            patches_path = os.path.join(args.dataset2_path, path)
        else:
            print(f"Skipping patient {patient}, dataset not found")
            continue  # Пропускаем пациента, если не нашли путь

        # Убираем повторяющиеся части пути (например, "TCGA-GBM_WSI/TCGA-GBM_WSI/")
        patches_path = patches_path.replace(f"{args.dataset1_path}/{os.path.basename(args.dataset1_path)}", f"{args.dataset1_path}")
        patches_path = patches_path.replace(f"{args.dataset2_path}/{os.path.basename(args.dataset2_path)}", f"{args.dataset2_path}")

        # Проверяем, является ли это файлом или директорией
        if os.path.isfile(patches_path):
            print(f"Found file for {patient}: {patches_path}")
            # Если это файл изображения (например, .svs), находим папку с патчами
            patches_folder = os.path.dirname(patches_path)  # Получаем папку, в которой находится .svs файл
            patches_path = os.path.join(patches_folder, 'patches')  # Должна быть папка "patches"

        # Проверим, что путь существует
        if not os.path.exists(patches_path):
            print(f"Warning: Path for {patient} does not exist: {patches_path}")
            continue

        # Извлекаем патчи
        patch_files = os.listdir(patches_path)
        if not patch_files:
            print(f"Warning: No patches found for patient {patient} in {patches_path}")
            continue

        print(f"Found {len(patch_files)} patches for patient {patient}")

        patches = [ToTensor()(np.array(Image.open(os.path.join(patches_path, patch)))) for patch in patch_files]
        patches = torch.stack(patches).to(device)

        with torch.no_grad():  # Выключаем вычисление градиентов
            embeddings = extractor(patches)  # Извлекаем эмбеддинги из модели

        # Выводим форму эмбеддингов
        print(f"Extracted embeddings shape for {patient}: {embeddings.shape}")
        
        # Сохраняем эмбеддинги в CSV файл
        temp_df = pd.DataFrame(embeddings.cpu().numpy())
        final_path = os.path.join(output_path, f'{patient}.csv')
        os.makedirs(os.path.dirname(final_path), exist_ok=True)  # Создаём необходимые директории
        temp_df.to_csv(final_path, index=False)
        
        # Записываем путь к файлу в маппинг
        print(f"Saving embeddings for {patient} to {final_path}")
        file_map[patient] = final_path  # Записываем путь к файлу в маппинг

    # Проверяем содержимое file_map
    print(f"File map contents: {file_map}")
    
    # Создаём финальную таблицу для маппинга
    print(f"Creating WSI mapping DataFrame with {len(file_map)} entries")
    WSI_mapping = pd.DataFrame([(k, v) for k, v in file_map.items()], columns=('submitter_id', 'WSI'))

    # Читаем датафрейм из переданного пути
    print(f"Reading dataframe from {args.data_path}")
    dataframe = pd.read_csv(args.data_path)  # Используем путь, переданный через argparse

    # Проверяем наличие столбца 'WSI' перед удалением
    if 'WSI' in dataframe.columns:
        dataframe.drop(columns=['WSI'], inplace=True)  # Удаляем старую колонку с WSI, если она есть
    else:
        print("'WSI' column not found, skipping drop.")

    # Мержим по идентификатору пациента
    print(f"Merging data with {len(WSI_mapping)} entries")
    dataframe = dataframe.merge(WSI_mapping, how='left', on='submitter_id')  # Мержим по идентификатору пациента

    # Сохраняем обновлённый датафрейм в директорию output_dir
    updated_dataframe_path = os.path.join(output_path, 'updated_dataframe_with_emb.csv')
    print(f"Saving updated dataframe to {updated_dataframe_path}")
    dataframe.to_csv(updated_dataframe_path, index=False)

if __name__ == "__main__":
    main()
