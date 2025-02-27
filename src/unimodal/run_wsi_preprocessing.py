import argparse
import os
import tqdm
import cv2
import numpy as np
import pyvips
import json
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Tuple, Dict
import random
import requests
import io
import matplotlib.pyplot as plt
from IPython import display


def request_file_info(data_type, base_path):
    """Request information about WSI files from GDC"""
    fields = [
        'file_name',
        'cases.submitter_id',
        'cases.samples.sample_type',
        'cases.project.project_id',
        'cases.project.primary_site',
    ]
    fields = ','.join(fields)
    files_endpt = 'https://api.gdc.cancer.gov/files'
    filters = {
        'op': 'and',
        'content': [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.primary_site",
                    "value": ["Brain"]
                }
            },
            {
                'op': 'in',
                'content': {
                    'field': 'files.experimental_strategy',
                    'value': [data_type]
                }
            }
        ]
    }
    params = {
        'filters': filters,
        'fields': fields,
        'format': 'TSV',
        'size': '200000'
    }
    response = requests.post(files_endpt, headers={'Content-Type': 'application/json'}, json=params)
    return pd.read_csv(io.StringIO(response.content.decode('utf-8')), sep=',')


def make_patient_file_map(df, base_dir):

    d = {}
    for _, row in df.iterrows():
        patient = row['cases.0.submitter_id']
        file_path = os.path.join(base_dir, row['id'], row['file_name'])
        if os.path.exists(file_path):
            if patient in d:
                if not isinstance(d[patient], tuple):
                    d[patient] = (d[patient], file_path)
                else:
                    d[patient] += (file_path,)
            else:
                d[patient] = file_path
    return d


def get_masked_hsv(patch: np.ndarray):

    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    mask = np.tile(patch[:, :, 1] > 150, (3, 1, 1)).transpose(1, 2, 0) * np.tile(patch[:, :, 2] < 150, (3, 1, 1)).transpose(1, 2, 0)
    return patch * mask




class PatchExtractor:
    def __init__(self, num_patches, patch_size, iterations, s_min: int = 150, v_max: int = 150):
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.iterations = iterations
        self.s_min = s_min
        self.v_max = v_max

    def patch_to_score(self, patch: np.ndarray):
        mask = np.tile(patch[:, :, 1] > self.s_min, (3, 1, 1)).transpose(1, 2, 0) * np.tile(patch[:, :, 2] < self.v_max, (3, 1, 1)).transpose(1, 2, 0)
        return (mask.sum(-1) / 3).sum()

    @staticmethod
    def _from_idx_to_row_col(idx: int, width: int) -> Tuple[int, int]:
        row = (idx // width)
        col = (idx % width)
        return (row, col)

    def __call__(self, slide, mask):
        patches_buffer = {}
        idx_buffer = {}
        factor = int(slide.width / mask.shape[1])
        delta = int(self.patch_size / factor)
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        kernel = torch.ones(1, 1, delta, delta)
        probabilities = F.conv2d(mask, kernel, stride=(delta, delta))
        probabilities = probabilities.squeeze()
        n_samples = torch.argwhere(probabilities).size(0) if torch.argwhere(probabilities).size(0) < self.iterations else self.iterations
        indexes = torch.multinomial(probabilities.view(-1), n_samples, replacement=False)
        for idx in indexes:
            patch, idx = self._from_idx_to_patch(slide, idx, probabilities.size(1))
            score = int(self.patch_to_score(cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)))
            patches_buffer[score] = patch
            patches_buffer = dict(sorted(patches_buffer.items(), key=lambda x: x[0], reverse=True))
            idx_buffer[score] = idx
            idx_buffer = dict(sorted(idx_buffer.items(), key=lambda x: x[0], reverse=True))
        return patches_buffer, idx_buffer

    def _from_idx_to_patch(self, slide, idx, width):
        idx = self._from_idx_to_row_col(idx, width)
        row = idx[0] * self.patch_size
        col = idx[1] * self.patch_size
        region = slide.crop(int(col), int(row), self.patch_size, self.patch_size)
        patch = np.ndarray(buffer=region.write_to_memory(),
                           dtype=np.uint8,
                           shape=(region.height, region.width, region.bands))
        return cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB), idx




def segment(
    img_rgba: np.ndarray,
    sthresh: int = 25,
    sthresh_up: int = 255,
    mthresh: int = 9,
    otsu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)  # Apply median blurring
    if otsu:
        _, img_otsu = cv2.threshold(
            img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY
        )
    else:
        _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)
    masked_image = cv2.bitwise_and(img, img, mask=img_otsu)
    return masked_image, img_otsu

import os
import pyvips
import tqdm
import cv2
import numpy as np
from PIL import Image

def create_thumbnail_and_mask(data_path, downscale_factor=6):
    
    subdirectories = os.listdir(data_path)
    for subdirectory in tqdm.tqdm(subdirectories):
        subdirectory_path = os.path.join(data_path, subdirectory)

        if os.path.isdir(subdirectory_path):  
            filenames = os.listdir(subdirectory_path)
        else:
            print(f"Skipped, not a directory: {subdirectory_path}")
            continue  
        
        # Select files with .svs or .tif extensions
        wsi_files = [f for f in filenames if f.endswith("svs") or f.endswith("tif")]

        if not wsi_files:
            print(f"No .svs or .tif files found in {subdirectory_path}")
            continue  # Skip this folder if there are no relevant files

        # Take the first file from the list
        wsi_filename = wsi_files[0]
        wsi_file_path = os.path.join(subdirectory_path, wsi_filename)

        # Check that it is a file and not a directory
        if not os.path.isfile(wsi_file_path):
            print(f"Skipped, {wsi_file_path} is not a file")
            continue

        try:
            # Load the image using pyvips
            slide = pyvips.Image.new_from_file(wsi_file_path)
        except Exception as e:
            print(f"Error loading image {wsi_file_path}: {e}")
            continue  # Skip this iteration in case of an error

        # Set scale for thumbnail
        if int(float(slide.get("aperio.AppMag"))) == 40:
            d = downscale_factor + 1
        else:
            d = downscale_factor

        # Create a thumbnail of the image
        thumbnail = pyvips.Image.thumbnail(
            wsi_file_path,
            slide.width / (2**d),
            height=slide.height / (2**d),
        ).numpy()

        # Convert to RGB color space
        thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGBA2RGB)
        thumbnail_hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)

        # Create a mask to remove felt-tip marks
        mask_hsv = np.tile(thumbnail_hsv[:, :, 1] < 160, (3, 1, 1)).transpose(1, 2, 0)
        thumbnail *= mask_hsv

        # Image segmentation
        masked_image, mask = segment(thumbnail)

        # Save results
        masked_image = Image.fromarray(masked_image).convert("RGB")
        masked_image.save(os.path.join(subdirectory_path, "thumbnail.jpg"))
        np.save(os.path.join(subdirectory_path, "mask.npy"), mask)




def sanity_check(base_path, num_patches=100):
    """Check the correctness of extracted patches"""
    for subdirectory in tqdm.tqdm(os.listdir(base_path)):
        subdirectory_path = os.path.join(base_path, subdirectory)
        
        # Skip if it's not a directory or if it's a 'logs' folder
        if not os.path.isdir(subdirectory_path) or subdirectory == 'logs':
            continue
        
        # Process only directories that are not named 'logs'
        patches_folder = os.path.join(subdirectory_path, 'patches')
        
        # Check if the patches folder exists
        if os.path.exists(patches_folder):
            if len(os.listdir(patches_folder)) != num_patches:
                print(f"Warning: Abnormal number of patches for {subdirectory}. Expected {num_patches}, found {len(os.listdir(patches_folder))}.")
            
            # Iterate through all files in the patches folder
            for patch_file in os.listdir(patches_folder):
                patch_path = os.path.join(patches_folder, patch_file)
                
                # Load each patch
                try:
                    patch = np.array(Image.open(patch_path))
                    # Check the patch size
                    if patch.shape != (256, 256, 3):
                        print(f"Abnormal patch size for {patch_file}. Expected (256, 256, 3), got {patch.shape}.")
                except Exception as e:
                    print(f"Error loading patch: {patch_path}. Error: {str(e)}")
        else:
            print(f"Patches folder not found: {patches_folder}")

def load_and_filter_wsi_data(mapping_file, dataframe, gbm_data_path, lgg_data_path):
    """Loading and filtering WSI data"""
    # Load your JSON file
    with open(mapping_file, 'r') as f:
        wsi_mapping = json.load(f)

    print("Columns in dataframe:", dataframe.columns)
    print("First rows of data:", dataframe.head())

    # If submitter_id exists in dataframe
    if 'submitter_id' in dataframe.columns:
        print(f"Unique submitter_ids in dataframe: {dataframe['submitter_id'].nunique()}")
    else:
        print("Error: Column 'submitter_id' not found in dataframe.")
    
    # Filter files from JSON by submitter_id
    # Use the correct path for GBM and LGG
    file_map_gbm = {
        k: v for k, v in wsi_mapping.items() 
        if k in dataframe['submitter_id'].values and os.path.exists(os.path.join(gbm_data_path, v.split('/')[-2], v.split('/')[-1]))
    }
    file_map_lgg = {
        k: v for k, v in wsi_mapping.items() 
        if k in dataframe['submitter_id'].values and os.path.exists(os.path.join(lgg_data_path, v.split('/')[-2], v.split('/')[-1]))
    }

    print(f"Number of files for GBM: {len(file_map_gbm)}")
    print(f"Number of files for LGG: {len(file_map_lgg)}")

    return {**file_map_gbm, **file_map_lgg}

def main(args):
    # Load the WSI mapping
    dataframe = pd.read_csv(args.wsi_file_path, sep=',')
    
    # Pass the correct paths for both datasets (GBM and LGG)
    file_map = load_and_filter_wsi_data(
        args.mapping_path, dataframe, args.gbm_data_path, args.lgg_data_path
    )
    print("Checking paths:", file_map)
    
    # Create thumbnails and masks for each patient
    create_thumbnail_and_mask(args.gbm_data_path, downscale_factor=args.downscale_factor)  # Replaced base_path with gbm_data_path
    create_thumbnail_and_mask(args.lgg_data_path, downscale_factor=args.downscale_factor)  # For LGG
    
    # Create and save the id2path dictionary
    id2path = {}
    for patient_id, path in file_map.items():
        if isinstance(path, tuple):
            # If multiple slides exist, display them with thumbnails
            n_slides = len(path)
            plt.figure(figsize=(12, 12))
            for i in range(n_slides):
                plt.subplot(1, n_slides, i+1)
                plt.axis('off')
                plt.title(pyvips.Image.new_from_file(path[i]).get('aperio.AppMag'))
                thumbnail_path = '/'.join(path[i].split('/')[:-1])
                thumbnail = np.array(Image.open(os.path.join(thumbnail_path, 'thumbnail.jpg')))
                plt.imshow(thumbnail)
            display.display(plt.gcf())
            display.clear_output(wait=True)
            plt.show()
            idx = int(input())
            id2path[patient_id] = path[idx]
        else:
            id2path[patient_id] = path

    # Overwrite the wsi_mapping.json file
    with open(args.mapping_path, 'w') as f:
        json.dump(id2path, f)

    # Read JSON data and merge it with dataframe
    with open(args.mapping_path, 'r') as f:
        wsi_mapping = json.load(f)

    WSI_mapping = pd.DataFrame([(k, v) for k, v in wsi_mapping.items()], columns=('submitter_id', 'WSI'))
    print(WSI_mapping)
    dataframe = dataframe.merge(WSI_mapping, how='left', on='submitter_id')
    dataframe.to_csv(args.wsi_file_path, index=False)

    # Process GBM and LGG data
    num_patches = args.num_patches
    patch_size = args.patch_size
    iterations = args.iterations

    # Process each patient's slide
    for patient, wsi_path in tqdm.tqdm(file_map.items()):
        # Check which dataset the file belongs to
        if 'GBM' in wsi_path:
            base_path = args.gbm_data_path
        else:
            base_path = args.lgg_data_path
        
        # Path to the .svs file
        wsi_full_path = os.path.join(base_path, *wsi_path.split('/')[1:])
        
        # Check if the .svs file exists
        if not os.path.exists(wsi_full_path):
            print(f"Error: File {wsi_full_path} does not exist.")
            continue
        
        print(f"Checking path: {wsi_full_path}")
        
        # Open the slide
        try:
            slide = pyvips.Image.new_from_file(wsi_full_path)  # Open slide file
            print(f"Image successfully loaded: {wsi_full_path}")
        except Exception as e:
            print(f"Error loading image: {wsi_full_path} \nError: {str(e)}")
            continue

        # Path to the folder for loading the mask
        folder_path = os.path.dirname(wsi_full_path)  # Path to the folder containing the .svs file
        mask_path = os.path.join(folder_path, 'mask.npy')
        
        # Load the mask
        try:
            mask = np.load(mask_path)
            print(f"Mask loaded: {mask_path}")
        except Exception as e:
            print(f"Error loading mask: {mask_path} \nError: {str(e)}")
            continue
        
        # Adjust patch extraction parameters based on magnification
        if int(float(slide.get('aperio.AppMag'))) == 40:
            extractor = PatchExtractor(num_patches=num_patches, patch_size=patch_size*2, iterations=iterations, s_min=130, v_max=170)
        else:
            extractor = PatchExtractor(num_patches=num_patches, patch_size=patch_size, iterations=iterations, s_min=130, v_max=170)
        
        # Extract patches
        patches, _ = extractor(slide, mask)

        # Sort patches by score
        if isinstance(patches, dict):
            patches = dict(sorted(patches.items(), key=lambda x: x[0], reverse=True))
        else:
            print("Warning: patches is not a dictionary, checking data structure.")
            print(f"Type of patches: {type(patches)}")
            # If patches is a tuple, assume the patches are in the first element
            patches = patches[0]  
            patches = dict(sorted(patches.items(), key=lambda x: x[0], reverse=True))

        selected_patches = {score: patch for score, patch in list(patches.items())[:num_patches]}
        
        # Create a folder for patches if it does not exist
        patches_folder = os.path.join(folder_path, 'patches')
        print(f"folder_path: {folder_path}")
        print(f"patches_folder: {patches_folder}")

        if not os.path.exists(patches_folder):
            os.makedirs(patches_folder)
        
        # Save patches
        for i, (score, patch) in enumerate(selected_patches.items()):
            patch = Image.fromarray(patch)
            patch.save(os.path.join(patches_folder, f'{int(i)}_{int(score)}.png'))
        
        # Perform a sanity check
        sanity_check(folder_path) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI patch extraction and thumbnail generation")
    parser.add_argument("--gbm_data_path", "-g", default="/mnt/public-datasets/drim/TCGA-GBM_WSI", help="Path to the GBM data folder")
    parser.add_argument("--lgg_data_path", "-l", default="/mnt/public-datasets/drim/wsi", help="Path to the LGG data folder")
    parser.add_argument("--mapping_path", "-m", default="/mtcp/src/data/wsi_mapping.json", help="Path to WSI mapping file")
    parser.add_argument("--num_patches", "-n", type=int, default=100, help="Number of patches to extracе")
    parser.add_argument("--patch_size", "-s", type=int, default=256, help="Size of the patches (256x256)")
    parser.add_argument("--iterations", "-i", type=int, default=1000, help="Number of iterations for patch extraction")
    parser.add_argument("--wsi_file_path", "-w", default="/mtcp/updated_utf8.csv", help="Path to the WSI files metadata")
    parser.add_argument("--downscale_factor", "-d", type=int, default=6, help="Downscale factor for thumbnail generation")

    args = parser.parse_args()
    main(args)
