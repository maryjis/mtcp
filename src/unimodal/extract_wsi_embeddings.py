import os
import json
import torch
import tqdm
import pyvips
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from typing import Tuple
import argparse

# Параметры командной строки
parser = argparse.ArgumentParser(description="Extract embeddings from WSI patches.")
parser.add_argument("--model_name", type=str, required=True, help="Name of the model (e.g., 'resnet34').")
parser.add_argument("--model_weights", type=str, required=True, help="Path to the model weights.")
parser.add_argument("--data_path", type=str, required=True, help="Path to the input WSI data.")
parser.add_argument("--output_path", type=str, required=True, help="Path to save the output embeddings.")
parser.add_argument("--patch_size", type=int, default=256, help="Size of the patches.")
parser.add_argument("--num_patches", type=int, default=100, help="Number of patches to process for each patient.")
args = parser.parse_args()

# Очистка состояния словаря модели
def clean_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'module.encoder' in key:
            new_state_dict[key[15:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

# Функция для загрузки модели
def load_model(model_name, model_weights):
    if model_name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=False)
    elif model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
    else:
        raise ValueError(f"Model {model_name} is not supported. Please choose a valid model.")
    
    model.fc = torch.nn.Identity()  # Убираем последнюю слой
    state_dict = clean_state_dict(torch.load(model_weights))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Класс для обработки патчей
class PatchDataset(Dataset):
    def __init__(self, patch_paths):
        self.patch_paths = patch_paths

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        patch_path = self.patch_paths[idx]
        patch = Image.open(patch_path)
        patch = ToTensor()(np.array(patch))
        return patch

# Извлечение эмбеддингов
def extract_embeddings(model, patch_paths, output_dir):
    dataset = PatchDataset(patch_paths)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    file_map = {}
    
    with torch.no_grad():
        for i, patches in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            embeddings = model(patches)
            for j, embedding in enumerate(embeddings):
                patient_id = os.path.basename(patch_paths[i * 32 + j]).split('.')[0]
                patient_output_path = os.path.join(output_dir, f"{patient_id}.csv")
                temp_df = pd.DataFrame(embedding.cpu().numpy()).transpose()
                temp_df.to_csv(patient_output_path, mode='a', header=not os.path.exists(patient_output_path), index=False)
                file_map[patient_id] = patient_output_path
                
    return file_map

# Основная функция
def process_wsi_data():
    model = load_model(args.model_name, args.model_weights)

    # Загрузка списка с путями WSI
    with open(os.path.join(args.data_path, 'src/data/wsi_mapping.json'), 'r') as f:
        mapping = json.load(f)

    # Создание папки для вывода, если её нет
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Обработка каждого пациента
    file_map = {}
    for patient, path in tqdm.tqdm(mapping.items(), total=len(mapping)):
        idx = os.path.basename(path).split('/')[-2]
        patches_path = os.path.join('/'.join(path.split('/')[:-1]), 'patches')
        patch_paths = [os.path.join(patches_path, patch) for patch in os.listdir(patches_path)]

        if len(patch_paths) > args.num_patches:
            patch_paths = patch_paths[:args.num_patches]

        patient_output_path = os.path.join(args.output_path, idx)
        if not os.path.exists(patient_output_path):
            os.makedirs(patient_output_path)
        
        file_map_patient = extract_embeddings(model, patch_paths, patient_output_path)
        file_map.update(file_map_patient)

    # Сохранение final mapping
    WSI_mapping = pd.DataFrame([(k, v) for k, v in file_map.items()], columns=('submitter_id', 'WSI'))
    WSI_mapping.to_csv(os.path.join(args.output_path, 'final_mapping.csv'), index=False)

if __name__ == "__main__":
    process_wsi_data()
