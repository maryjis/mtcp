import random, os
import numpy as np
import torch
from monai.utils import set_determinism
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union
import wandb
from omegaconf import DictConfig, OmegaConf

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    set_determinism(seed=seed, additional_settings=None)
    
    
def load_splits(data_path: Path, fold_ind : int, remove_nan_column : str) -> Dict[str,pd.DataFrame]:
    if data_path.exists():
        
        dataset = pd.read_csv(data_path)
        if remove_nan_column:
            dataset =dataset.loc[dataset[remove_nan_column].notnull()]
        
        dataset_test = dataset.loc[dataset.group =='test']
        dataset_train_val = dataset.loc[dataset.group =='train']
        
        if fold_ind in dataset_train_val.splits.unique():
            dataset_val = dataset_train_val[dataset_train_val.splits==fold_ind]
            dataset_train = dataset_train_val[dataset_train_val.splits!=fold_ind]
        else:
            raise Exception(f"Such fold index {fold_ind} doesn't exist. Check {data_path} file. ")
        return {"train" :dataset_train.reset_index(drop=True), 
                "val" : dataset_val.reset_index(drop=True), 
                "test" : dataset_test.reset_index(drop=True)}
    else:
        raise Exception(f"Dataset file {data_path} didn't found.")
    
def init_wandb_logging(cfg : DictConfig):
      print(cfg)
      cfg_dict = OmegaConf.to_container(cfg)
      wandb.init(
        project=cfg.wandb_project,
        config=cfg_dict)
      
      wandb.define_metric("train/*", step_metric="epoch")
      wandb.define_metric("valid/*", step_metric="epoch")
      
def agg_fold_metrics(lst: list[dict[str, float]]):
    """Compute mean, min, max, std from cross validation metrics"""
    keys = lst[0].keys()
    res = {}
    for k in keys:
        res[k] = compute_stats([dct[k] for dct in lst])
    return res


def compute_stats(lst: list[float]) -> dict[str, np.ndarray]:
    """Compute some stats from a list of floats"""
    arr = np.array(lst)
    return {"mean": arr.mean(), "std": arr.std(), "min": arr.min(), "max": arr.max()}      