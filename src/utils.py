import random, os
import numpy as np
import torch
from monai.utils import set_determinism
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn
import math
from math import ceil
from einops import rearrange, reduce


MODALITY_TO_COLUMN_MAP ={"rna" : "RNA", "mri" : "MRI", "dnam" : "DNAm", "wsi": "WSI_initial", "cnv": "CNV"}

# Components for transformer pool
def exists(val: any) -> bool:
    """Checks if value exists (not None)."""
    return val is not None


def moore_penrose_iter_pinv(x: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """
    Iteratively computes the Moore-Penrose pseudoinverse matrix.
    
    Args:
        x: Input tensor
        iters: Number of iterations for approximation
        
    Returns:
        Pseudoinverse matrix
    """
    device = x.device
    
    # Normalization for numerical stability
    abs_x = torch.abs(x)
    norm_factor = torch.max(abs_x.sum(dim=-1)) * torch.max(abs_x.sum(dim=-2))
    z = rearrange(x, "... i j -> ... j i") / norm_factor
    
    # Identity matrix for computation
    I = torch.eye(x.shape[-1], device=device).unsqueeze(0)
    
    # Iterative computation of pseudoinverse matrix
    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))
    
    return z


class NystromAttention(nn.Module):
    """
    Efficient implementation of attention with Nyström approximation.
    Reduces complexity from O(n²) to O(n*m), where m is the number of landmarks.
    """
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        num_landmarks: int = 256,
        pinv_iterations: int = 6,
        residual: bool = True,
        residual_conv_kernel: int = 33,
        eps: float = 1e-8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head
        
        # Attention parameters
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.heads = heads
        self.scale = dim_head**-0.5
        
        # Layers
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Residual convolutional connection
        self.residual = residual
        if residual:
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(
                heads, 
                heads, 
                kernel_size=(residual_conv_kernel, 1),
                padding=(padding, 0),
                groups=heads,
                bias=False
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attn: bool = False):
        b, n, _ = x.shape
        h, m, iters, eps = self.heads, self.num_landmarks, self.pinv_iterations, self.eps
        
        # Padding for even division of sequence into landmarks
        if n % m != 0:
            padding = m - (n % m)
            x = torch.nn.functional.pad(x, (0, 0, 0, padding), value=0)
            if exists(mask):
                mask = torch.nn.functional.pad(mask, (0, padding), value=False)
        
        # Get Q, K, V from input
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        
        # Apply mask if it exists
        if exists(mask):
            mask = mask.unsqueeze(1)  # b, 1, n
            q, k, v = map(lambda t: t * mask.unsqueeze(-1), (q, k, v))
        
        # Scale queries
        q = q * self.scale
        
        # Compute landmarks by aggregation
        l = ceil(n / m)
        q_landmarks = reduce(q, "b h (n l) d -> b h n d", "mean", l=l)
        k_landmarks = reduce(k, "b h (n l) d -> b h n d", "mean", l=l)
        
        # Compute similarity matrices
        sim1 = torch.einsum("b h i d, b h j d -> b h i j", q, k_landmarks)  # (b, h, n, m)
        sim2 = torch.einsum("b h i d, b h j d -> b h i j", q_landmarks, k_landmarks)  # (b, h, m, m)
        sim3 = torch.einsum("b h i d, b h j d -> b h i j", q_landmarks, k)  # (b, h, m, n)
        
        # Apply mask to similarity computations if it exists
        if exists(mask):
            # Create mask for landmarks
            mask_landmarks = reduce(mask, "b 1 (n l) -> b 1 n", "any", l=l)
            
            # Calculate value for filling masked positions
            mask_value = -torch.finfo(q.dtype).max
            
            # Apply masks to similarity matrices
            sim1.masked_fill_(~(mask.unsqueeze(-1) * mask_landmarks.unsqueeze(-2)), mask_value)
            sim2.masked_fill_(~(mask_landmarks.unsqueeze(-1) * mask_landmarks.unsqueeze(-2)), mask_value)
            sim3.masked_fill_(~(mask_landmarks.unsqueeze(-1) * mask.unsqueeze(-2)), mask_value)
        
        # Get attention weight coefficients
        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        
        # Compute pseudoinverse matrix
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)
        
        # Compute attention output
        out = (attn1 @ attn2_inv) @ (attn3 @ v)
        
        # Add residual connection through convolution if enabled
        if self.residual:
            out = out + self.res_conv(v)
        
        # Combine attention heads
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        
        # Trim to original length
        out = out[:, :n]
        
        # Return output and, if needed, attention weights
        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn
            
        return out


class TransLayer(nn.Module):
    """
    Transformer layer with normalization and Nyström attention.
    """
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of iterations for approximating pseudoinverse
            residual=True,  # whether to use residual connection through values
            dropout=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply attention with normalization and residual connection
        return x + self.attn(self.norm(x))


import torch
import torch.nn as nn
from timm.models.layers import DropPath

class PPEG(nn.Module):
    """
    Positional Encoding with Grouped Convolutions + Residual Fusion.
    Inspired by CPVT, ConvNeXt, and CoAtNet.
    """
    def __init__(self, dim: int = 512, drop_path: float = 0.1):
        super().__init__()
        
        def depthwise_block(kernel_size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=dim),
                nn.BatchNorm2d(dim),
                nn.GELU()
            )
        
        # Multi-scale depthwise convolutions
        self.proj = depthwise_block(7)
        self.proj1 = depthwise_block(5)
        self.proj2 = depthwise_block(3)

        # Learnable fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # DropPath regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Normalization for cls_token
        self.cls_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        assert N == 1 + H * W, f"Expected sequence length {1 + H*W}, got {N}"
        
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).reshape(B, C, H, W)

        # Multi-scale depthwise convolutions
        x1 = self.proj(cnn_feat)
        x2 = self.proj1(cnn_feat)
        x3 = self.proj2(cnn_feat)

        # Concatenate and fuse
        x_cat = torch.cat([cnn_feat, x1, x2, x3], dim=1)  # [B, 4C, H, W]
        x_fused = self.fusion(x_cat)
        x_fused = self.drop_path(x_fused)

        # Reshape back to sequence
        x_flat = x_fused.flatten(2).transpose(1, 2)  # [B, H*W, C]
        cls_token = self.cls_norm(cls_token)
        x_out = torch.cat((cls_token.unsqueeze(1), x_flat), dim=1)
        return x_out


def get_config_mode(config_path, base_path=None):
    if base_path is not None:
        config_path = os.path.join(base_path, config_path)
    mode = config_path.split(".")[-2]
    if mode not in ["done", "in_progress"]: 
        return None
    return mode

def append_config_mode(config_path, mode, base_path=None):
    if base_path is not None:
        config_path = os.path.join(base_path, config_path)

    if get_config_mode(config_path) != mode:
        config_extension = config_path.split(".")[-1]
        config_path_without_extension = ".".join(config_path.split(".")[:-1]) if get_config_mode(config_path) is None else ".".join(config_path.split(".")[:-2])
        new_config_path = config_path_without_extension + "." + mode + "." + config_extension
        os.rename(
            config_path, 
            new_config_path
        )
        return new_config_path
    else:
        return config_path

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
                max_samples_per_split : int = None, multimodal_intersection_test:bool = False,
                modalities: List[str] = None, project_ids: List[str] = None ) -> Dict[str,pd.DataFrame]:
    '''
    Use 'group' column to split data into train_validation/test sets
    Loads and splits a dataset from a CSV file into train/validation/test sets
    Handles cross-validation folds by separating data based on a 'splits' column
    Can remove rows with NaN values in a specified column
    Returns a dictionary with three DataFrames: train, validation, and test sets
    '''
    
    if data_path.exists():
        
        dataset = pd.read_csv(data_path)
        print("Dataset shape before: ", dataset.shape)
        if project_ids:
            dataset =dataset.loc[dataset["project_id"].isin(project_ids)]
        print("Dataset shape before: ", dataset.shape)
        if remove_nan_column:
            dataset =dataset.loc[dataset[remove_nan_column].notnull()]
        print("Dataset shape after: ", dataset.shape)
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
