import random, os
import numpy as np
import torch
from monai.utils import set_determinism
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn

def print_vit_for_pretrain_sizes(model):
    print(f"Total number of parameters : {count_parameters(model)}")
    print(f"--ViT: {count_parameters(model.vit)}")
    print(f"----Embeddings: {count_parameters(model.vit.embeddings)}")
    print(f"------Patch embeddings: {count_parameters(model.vit.embeddings.patch_embeddings)}")
    print(f"------Position embeddings: {count_parameters(model.vit.embeddings.position_embeddings)}")
    print(f"------Class token: {count_parameters(model.vit.embeddings.cls_token)}")
    print(f"----Encoder: {count_parameters(model.vit.encoder)}")
    print(f"----Layer norm: {count_parameters(model.vit.layernorm)}")

    print(f"--Decoder: {count_parameters(model.decoder)}")
    print(f"----Decoder embed: {count_parameters(model.decoder.decoder_embed)}")
    print(f"----Mask token: {count_parameters(model.decoder.mask_token)}")
    print(f"----Decoder pos embed: {count_parameters(model.decoder.decoder_pos_embed)}")
    print(f"----Decoder layers: {count_parameters(model.decoder.decoder_layers)}")
    print(f"----Decoder norm: {count_parameters(model.decoder.decoder_norm)}")
    print(f"----Decoder pred: {count_parameters(model.decoder.decoder_pred)}")

def count_parameters(model):
    if isinstance(model, nn.Parameter):
        return model.numel()
    elif isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

def check_dir_exists(save_path: str):
    dir_to_save = os.path.join(*save_path.split(os.sep)[:-1])
    if not os.path.exists(dir_to_save): os.makedirs(dir_to_save, exist_ok=True)

def clean_state_dict(state_dict: dict) -> dict:
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
    '''
    Use 'group' column to split data into train_validation/test sets
    Loads and splits a dataset from a CSV file into train/validation/test sets
    Handles cross-validation folds by separating data based on a 'splits' column
    Can remove rows with NaN values in a specified column
    Returns a dictionary with three DataFrames: train, validation, and test sets
    '''
    
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
        config=cfg_dict,
        name=cfg.wandb_run_name
    )
      
    #   wandb.define_metric("train/*", step_metric="epoch")
    #   wandb.define_metric("valid/*", step_metric="epoch")
      
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