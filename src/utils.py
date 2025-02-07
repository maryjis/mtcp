import random, os
import numpy as np
import torch
from monai.utils import set_determinism
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn


MODALITY_TO_COLUMN_MAP ={"rna" : "RNA", "mri" : "MRI", "dnam" : "DNAm"}

def trace_handler(p, sort_by_keyword="self_cpu_time_total", row_limit=10, phase="train", is_print=True):
    if is_print: print(p.key_averages().table(sort_by=sort_by_keyword, row_limit=row_limit))
    base_path = "outputs/traces"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    p.export_chrome_trace(f"{base_path}/trace_{p.step_num}_{phase}.json")


def print_vit_sizes(model):
    print(f"Total number of parameters : {count_parameters(model)}")

    if hasattr(model, "vit"):
        print(f"--ViT: {count_parameters(model.vit)}")
        print(f"----Embeddings: {count_parameters(model.vit.embeddings)}")
        print(f"------Patch embeddings: {count_parameters(model.vit.embeddings.patch_embeddings)}")
        print(f"------Position embeddings: {count_parameters(model.vit.embeddings.position_embeddings)}")
        print(f"------Class token: {count_parameters(model.vit.embeddings.cls_token)}")
        print(f"----Encoder: {count_parameters(model.vit.encoder)}")
        print(f"----Layer norm: {count_parameters(model.vit.layernorm)}")

    if hasattr(model, "decoder"):
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
    
    
def load_splits(data_path: Path, fold_ind : int, remove_nan_column : str, 
                max_samples_per_split : int = None, multimodal_intersection_test:bool = False, modalities: List[str] = None ) -> Dict[str,pd.DataFrame]:
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

        if max_samples_per_split:
            dataset_train = dataset_train.sample(max_samples_per_split)
            dataset_val = dataset_val.sample(max_samples_per_split)
            dataset_test = dataset_test.sample(max_samples_per_split)
            print(f"WARNING: max_samples_per_split={max_samples_per_split} is set.")
        
        if multimodal_intersection_test:
            dataset_intersection_test = dataset_test.copy()
            for modality in modalities:
                if modality == "clinical":
                    continue
                dataset_intersection_test =dataset_intersection_test.loc[dataset_intersection_test[MODALITY_TO_COLUMN_MAP[modality]].notnull()]
            print("Multimodal intersection test: ", dataset_intersection_test.shape)
            return {"train" :dataset_train.reset_index(drop=True), 
                    "val" : dataset_val.reset_index(drop=True), 
                    "test" : dataset_test.reset_index(drop=True),
                    "test_intersection" : dataset_intersection_test.reset_index(drop=True)}
        else:    
            return {"train" :dataset_train.reset_index(drop=True), 
                    "val" : dataset_val.reset_index(drop=True), 
                    "test" : dataset_test.reset_index(drop=True)}
    else:
        raise Exception(f"Dataset file {data_path} didn't found.")
    
def init_wandb_logging(cfg : DictConfig):
    print(cfg)
    cfg_dict = OmegaConf.to_container(cfg)
    wandb.init(
        project=cfg.base.log.wandb_project,
        config=cfg_dict,
        name=cfg.base.log.wandb_run_name
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

def add_model_paths_to_config(cfg : DictConfig, fold_ind: int):
    
            if cfg.model.get("rna_model", False):
                with open_dict(cfg):
                    print("Model path", f"outputs/models/{cfg.model.rna_model.pretrained_model_name}_split_{fold_ind}.pth")
                    cfg.model.rna_model.pretrained_model_path = f"outputs/models/{cfg.model.rna_model.pretrained_model_name}_split_{fold_ind}.pth"
            
            if cfg.model.get("mri_model", False):
                with open_dict(cfg):
                    print("Model path", f"outputs/models/{cfg.model.mri_model.pretrained_model_name}_split_{fold_ind}.pth")
                    cfg.model.mri_model.pretrained_model_path = f"outputs/models/{cfg.model.mri_model.pretrained_model_name}_split_{fold_ind}.pth"
            
            if cfg.model.get("dnam_model", False):
                with open_dict(cfg):
                    print("Model path", f"outputs/models/{cfg.model.dnam_model.pretrained_model_name}_split_{fold_ind}.pth")
                    cfg.model.dnam_model.pretrained_model_path = f"outputs/models/{cfg.model.dnam_model.pretrained_model_name}_split_{fold_ind}.pth"
            
            if cfg.model.get("missing_modalities_strategy", False)=="decoder":
                with open_dict(cfg):
                    print("Model path", f"outputs/models/{cfg.model.mm_pretrained_model_name}_split_{fold_ind}.pth")
                    cfg.model.mm_pretrained_model_path = f"outputs/models/{cfg.model.mm_pretrained_model_name}_split_{fold_ind}.pth" 
                    if cfg.model.mm_decoder_config.get("mri_model", False):
                        with open_dict(cfg):
                            print("Model path", f"outputs/models/{cfg.model.mri_model.pretrained_model_name}_split_{fold_ind}.pth")
                            cfg.model.mm_decoder_config.mri_model.pretrained_model_path = f"outputs/models/{cfg.model.mri_model.pretrained_model_name}_split_{fold_ind}.pth"
                            print(cfg.model.mm_decoder_config.mri_model.pretrained_model_path)
                print("cfg.model.mm_pretrained_model_path", cfg.model.mm_pretrained_model_path)
            return cfg       