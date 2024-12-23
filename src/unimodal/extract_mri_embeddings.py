import pandas as pd
import argparse
import torch
import csv
import tqdm
import os

from src.unimodal.mri.transforms import tumor_transforms
from src.unimodal.mri.datasets import MRIProcessor
from src.unimodal.mri.models import MRIEncoder
from src.utils import clean_state_dict
from src.logger import logger

parser = argparse.ArgumentParser()
parser.add_argument("--skip_existing_embeddings", type=bool, default=True)
parser.add_argument("--data_path", type=str, default="src/data/dataset.csv")
parser.add_argument("--model_path", type=str, default="src/data/models/t1c-t2f_tumorTrue.pth")
args = parser.parse_args()

# Load the data
data = pd.read_csv(args.data_path)
encoder = MRIEncoder(2, 512, False)
encoder.load_state_dict(
    clean_state_dict(torch.load(args.model_path)), strict=False
)
encoder.eval()
for mri_path in tqdm.tqdm(data.MRI):
    try:
        if pd.isna(mri_path):
            continue
        process = MRIProcessor(
            mri_path,
            tumor_centered=True,
            transform=tumor_transforms,
            modalities=["t1c", "t2f"],
            size=(64, 64, 64),
        )
        mri = process.process().unsqueeze(0)
        with torch.no_grad():
            embedding = encoder(mri)

        # Save the embedding
        if os.path.exists(os.path.join(mri_path, "embedding.csv")):
            if args.skip_existing_embeddings:
                continue
            # logger.info(f"Embedding for {mri_path} and was overritten")

        with open(os.path.join(mri_path, "embedding.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow([str(i) for i in range(512)])
            writer.writerow(embedding.squeeze().numpy().tolist())
            
    except KeyboardInterrupt:
        break
    except Exception as e:
        logger.info(f"ERROR for {mri_path}: {e}")
