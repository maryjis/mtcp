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

def clean_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'module.encoder' in key:
            new_state_dict[key[15:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Extract embeddings from WSI patches.")
    
    parser.add_argument('--model_path', type=str, default="/home/a.beliaeva/mtcp/src/outputs/models/wsi_encoder.pth", help="Path to the pretrained model weights.")
    parser.add_argument('--mapping_path', type=str, default="/home/belyaeva.a/mtcp/src/data/wsi_mapping.json", help="Path to the JSON mapping file for WSI images.")
    parser.add_argument('--output_dir', type=str, default="/home/a.beliaeva/mtcp/src/outputs/embeddings_wsi", help="Directory to save extracted embeddings.")
    parser.add_argument('--dataset1_path', type=str, default="/mnt/public-datasets/drim/TCGA-GBM_WSI", help="Path to the first dataset directory.")
    parser.add_argument('--dataset2_path', type=str, default="/mnt/public-datasets/drim/wsi", help="Path to the second dataset directory.")
    parser.add_argument("--data_path", type=str, default="/home/belyaeva.a/mtcp/src/data/dataset.csv", help="Path to the input dataframe.")
    parser.add_argument('--cuda', action='store_true', help="Use CUDA if available.")

    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    print(f"Loading model from {args.model_path}")
    extractor = torchvision.models.resnet34(pretrained=False)
    extractor.fc = torch.nn.Identity() 
    state_dict = clean_state_dict(torch.load(args.model_path))  
    extractor.load_state_dict(state_dict, strict=False)
    extractor.to(device)
    extractor.eval()  


    print(f"Loading mapping from {args.mapping_path}")
    with open(args.mapping_path, 'r') as f:
        mapping = json.load(f)
    

    print(f"Loaded mapping with {len(mapping)} patients.")
    

    output_path = os.path.join(args.output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_map = {}
    print("Processing patients...")
    for patient, path in tqdm.tqdm(mapping.items(), total=len(mapping)):
        print(f"Processing patient: {patient}")

        
        if "TCGA-GBM_WSI" in path:
            patches_path = os.path.join(args.dataset1_path, path)
        elif "wsi" in path:  # Для второго датасета
            patches_path = os.path.join(args.dataset2_path, path)
        else:
            print(f"Skipping patient {patient}, dataset not found")
            continue  

        
        patches_path = patches_path.replace(f"{args.dataset1_path}/{os.path.basename(args.dataset1_path)}", f"{args.dataset1_path}")
        patches_path = patches_path.replace(f"{args.dataset2_path}/{os.path.basename(args.dataset2_path)}", f"{args.dataset2_path}")

       
        if os.path.isfile(patches_path):
            print(f"Found file for {patient}: {patches_path}")
            
            patches_folder = os.path.dirname(patches_path) 
            patches_path = os.path.join(patches_folder, 'patches') 

        
        if not os.path.exists(patches_path):
            print(f"Warning: Path for {patient} does not exist: {patches_path}")
            continue

        
        patch_files = os.listdir(patches_path)
        if not patch_files:
            print(f"Warning: No patches found for patient {patient} in {patches_path}")
            continue

        print(f"Found {len(patch_files)} patches for patient {patient}")

        patches = [ToTensor()(np.array(Image.open(os.path.join(patches_path, patch)))) for patch in patch_files]
        patches = torch.stack(patches).to(device)

        with torch.no_grad(): 
            embeddings = extractor(patches)  

        
        print(f"Extracted embeddings shape for {patient}: {embeddings.shape}")
        
        
        temp_df = pd.DataFrame(embeddings.cpu().numpy())
        final_path = os.path.join(output_path, f'{patient}.csv')
        os.makedirs(os.path.dirname(final_path), exist_ok=True)  
        temp_df.to_csv(final_path, index=False)
        
        print(f"Saving embeddings for {patient} to {final_path}")
        file_map[patient] = final_path 

    
    print(f"File map contents: {file_map}")
    
    
    print(f"Creating WSI mapping DataFrame with {len(file_map)} entries")
    WSI_mapping = pd.DataFrame([(k, v) for k, v in file_map.items()], columns=('submitter_id', 'WSI'))

    
    print(f"Reading dataframe from {args.data_path}")
    dataframe = pd.read_csv(args.data_path)  
    
    if 'WSI' in dataframe.columns:
        dataframe.rename(columns={'WSI': 'WSI_initial'}, inplace=True)
 
    else:
        print("'WSI' column not found, skipping drop.")

    print(f"Merging data with {len(WSI_mapping)} entries")
    dataframe = dataframe.merge(WSI_mapping, how='left', on='submitter_id') 

   
    updated_dataframe_path = os.path.join(output_path, 'updated_dataframe_with_emb.csv')
    print(f"Saving updated dataframe to {updated_dataframe_path}")
    dataframe.to_csv(updated_dataframe_path, index=False)

if __name__ == "__main__":
    main()
