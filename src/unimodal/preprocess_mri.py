import pandas as pd
import argparse
import torch
import tqdm
import os

from src.unimodal.mri.transforms import tumor_transforms
from src.unimodal.mri.datasets import MRIProcessor
from src.logger import logger
from src.unimodal.mri.utils import get_patients_from_BraTS

parser = argparse.ArgumentParser()
parser.add_argument("--skip_existing_roi", type=bool, default=True)
parser.add_argument("--data_path", type=str, default="/data/BraTS_2023/MRI")
parser.add_argument("--modalities", type=str, nargs="+", default=["t1c"]) #["t1c", "t2f"]
parser.add_argument("--roi_name", type=str, default="roi")
args = parser.parse_args()

# Load the data

patients = get_patients_from_BraTS(args.data_path, args.modalities, with_mask=True, df_with_test=None)

for mri_path in tqdm.tqdm(patients):
    try:
        if pd.isna(mri_path):
            continue

        mri_path = os.path.join(args.data_path, mri_path)
        size = (64, 64, 64)
        process = MRIProcessor(
            mri_path,
            tumor_centered=True,
            transform=tumor_transforms,
            modalities=args.modalities,
            size=size,
        )
        mri = process.process()

        # Save the roi
        roi_name = os.path.join(
            mri_path, 
            f"{args.roi_name}_{'_'.join(args.modalities)}_{'_'.join(list(map(str, size)))}.pt"
        )
        if os.path.exists(roi_name):
            if args.skip_existing_roi:
                logger.info(f"Embedding for {mri_path} already exists")
                continue
        torch.save(mri, roi_name)
            
    except KeyboardInterrupt:
        break
    except Exception as e:
        logger.info(f"ERROR for {mri_path}: {e}")
