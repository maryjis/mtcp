import os
import numpy as np
from typing import Dict, List, Tuple, Union
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from src.unimodal.rna.dataset import RNADataset, RNASurvivalDataset
from src.unimodal.dna.datasets import DNAmDataset, DNAmSurvivalDataset
from src.unimodal.dna.models import initialise_dnam_mae_model,  initialise_dnam_model, DNAmMAEForPreTraining
from src.unimodal.dna.preprocessor import DNAmPreprocessor
from src.unimodal.wsi.datasets import WSIDataset, WSIDataset_patches, SurvivalWSIDataset
from src.unimodal.mri.datasets import SurvivalMRIDataset, MRIEmbeddingDataset, MRIDataset, MRISurvivalDataset

from src.unimodal.rna.preprocessor import RNAPreprocessor
from src.preprocessor import BaseUnimodalPreprocessor
from src.unimodal.rna.transforms import base_transforms, padded_transforms_with_scaling
from torch.utils.data import Dataset, DataLoader
from pycox.models.loss import NLLLogistiHazardLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR   
from src.unimodal.rna.encoder import initialise_rna_model
from src.unimodal.mri.models import MRIEmbeddingEncoder
from src.unimodal.wsi.models import WSIEncoder
from src.unimodal.rna.mae import initialise_rna_mae_model
import torch
from ..evaluation import compute_survival_metrics
import wandb
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from src.unimodal.rna.mae import RnaMAEForPreTraining
from src.unimodal.mri.mae import MriMAEForPreTraining, MriMaeSurvivalModel
from src.unimodal.mri.noise import MriNoiseForPreTraining, MriNoiseSurvivalModel
from src.unimodal.wsi.mae import WsiMAEForPreTraining, WsiMaeSurvivalModel
from src.utils import check_dir_exists, count_parameters, print_vit_sizes

from src.unimodal.clinical.datasets import ClinicalDataset, ClinicalSurvivalDataset
from src.unimodal.clinical.preprocessor import ClinicalPreprocessor
from src.unimodal.clinical.transforms import base_scaling

from tqdm.auto import tqdm
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
from src.unimodal.rna.transforms import UpperQuartileNormalizer
from src.unimodal.mri.transforms import get_basic_tumor_transforms
from src.unimodal.wsi.transforms import contrastive_wsi_transforms
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from src.utils import trace_handler
from functools import partial
from src.unimodal.dna.transforms import padded_transforms_simple
from src.early_stopper import EarlyStopper
import math

class Trainer(object):
    def __init__(self, splits: Dict[str,pd.DataFrame], cfg: DictConfig):
        self.cfg = cfg
        self.preproc = self.initialise_preprocessing(splits, self.cfg.base.modalities[0])
 
    def initialise_preprocessing(self, splits, modality):
        
        if modality=="rna":
            
            if self.cfg.data.rna.scaling_method in ["QuantileTransformer", "StandardScaler", "MinMaxScaler"]:
                scaling_method = getattr(__import__('sklearn.preprocessing', fromlist=[self.cfg.data.rna.scaling_method]), self.cfg.data.rna.scaling_method)
            elif self.cfg.data.rna.scaling_method=="UpperQuartileNormalizer":
                 scaling_method = UpperQuartileNormalizer 
            else:
                 scaling_method = None
            print("Scaling method: ", scaling_method)
            preproc = RNAPreprocessor(splits["train"], self.cfg.data.rna.rna_dataset_path, self.cfg.base.n_intervals, scaling_method, 
                                          self.cfg.data.rna.scaling_params, self.cfg.data.rna.var_threshold,
                                          self.cfg.data.rna.is_cluster_genes , self.cfg.data.rna.clustering_threshold, self.cfg.data.rna.get("is_hierarchical_clusters", False))
            preproc.fit()
            return preproc

        elif modality == "mri":
            preproc = None
            if self.cfg.base.strategy == "survival": #labels are not used for pre-training
                preproc = BaseUnimodalPreprocessor(splits["train"], self.cfg.base.n_intervals)
                preproc.fit()
            return preproc
        
        elif modality == "dnam":
            scaling_method = None
            if self.cfg.data.dnam.scaling_method in ["QuantileTransformer", "StandardScaler", "MinMaxScaler"]:
                scaling_method = getattr(__import__('sklearn.preprocessing', fromlist=[self.cfg.data.dnam.scaling_method]), self.cfg.data.dnam.scaling_method)
            preproc = DNAmPreprocessor(splits["train"], self.cfg.data.dnam.dnam_dataset_path,self.cfg.base.n_intervals, 
                                           self.cfg.data.dnam.var_threshold,
                                          self.cfg.data.dnam.is_cluster_genes , self.cfg.data.dnam.clustering_threshold,
                                          self.cfg.data.dnam.get("is_hierarchical_clusters", False),  scaling_method, self.cfg.data.dnam.scaling_params)
            preproc.fit()
            return preproc
        
        elif modality == "wsi":
            preproc = None
            if self.cfg.base.strategy == "survival": #labels are not used for pre-training
                preproc = BaseUnimodalPreprocessor(splits["train"], self.cfg.base.n_intervals)
                preproc.fit()
            return preproc
        
        elif modality == "clinical":
            preproc = ClinicalPreprocessor(splits["train"], 
                                          self.cfg.data.clinical.dataset_path,
                                          self.cfg.data.clinical.selected_columns, 
                                          self.cfg.base.n_intervals,
                                          StandardScaler, 
                                          {})
            preproc.fit()
            return preproc

        else:
            raise NotImplementedError("Exist only for rna and mri and wsi. Initialising preprocessing for other modalities aren't declared")    
                
                
    def train(self, fold_ind : int):
        if self.cfg.base.get("early_stopping", None) is not None:
            self.early_stopper = EarlyStopper(**self.cfg.base.early_stopping.early_stopper)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            schedule=schedule(
                skip_first = 1 if self.cfg.base.get("profiling", None) is not None else 2147483647, #we want record only training part of epoch
                wait = 0,
                warmup = 1,
                active = 1,
                repeat = 1
            ),
            on_trace_ready=partial(
                trace_handler, 
                sort_by_keyword=self.cfg.base.profiling.sort_by_keyword if self.cfg.base.get("profiling", None) is not None else None, 
                phase="train",
                is_print=self.cfg.base.profiling.is_print if self.cfg.base.get("profiling", None) is not None else None
            )
        ) as prof:
            min_val = 1000000
            for epoch in tqdm(range(self.cfg.base.n_epochs)):
                print("Train...")
                
                self.model.train()
                train_metrics = self.__loop__("train",fold_ind, self.dataloaders['train'], self.cfg.base.device)
                train_metrics.update({"epoch": epoch})
                if self.cfg.base.log.logging:
                    wandb.log({f"train/fold_{fold_ind}/{key}" : value for key, value in train_metrics.items()})

                prof.step()
                
                print("Val...")
                self.model.eval()
                with torch.no_grad():    
                    val_metrics = self.__loop__("val",fold_ind, self.dataloaders['val'], self.cfg.base.device)
                val_metrics.update({"epoch": epoch})    
                if self.cfg.base.log.logging:
                    wandb.log({f"val/fold_{fold_ind}/{key}" : value for key, value in val_metrics.items()})

                with torch.no_grad():    
                    test_metrics = self.__loop__("test",fold_ind, self.dataloaders['test'], self.cfg.base.device)
                test_metrics.update({"epoch": epoch})    
                if self.cfg.base.log.logging:
                    wandb.log({f"test/fold_{fold_ind}/{key}" : value for key, value in test_metrics.items()})
                    
                prof.step()
  
                if self.cfg.base.get("early_stopping", None) is not None:
                    print("Early stopping: ")
                    if self.early_stopper.early_stop(val_metrics[self.cfg.base.early_stopping.value_to_track]):
                        break
                    if val_metrics[self.cfg.base.early_stopping.value_to_track]<= min_val:
                        print(f"Archive min error on validation  {val_metrics[self.cfg.base.early_stopping.value_to_track]} , saving model...")
                        min_val = val_metrics[self.cfg.base.early_stopping.value_to_track]
                        check_dir_exists(self.cfg.base.save_path)
                        torch.save(self.model.state_dict(), self.cfg.base.save_path)
                else:
                    check_dir_exists(self.cfg.base.save_path)
                    torch.save(self.model.state_dict(), self.cfg.base.save_path)

        return val_metrics
    
    def evaluate(self, fold_ind : int):
        print("Test...")

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            schedule=schedule(
                skip_first = 0 if self.cfg.base.get("profiling", None) is not None else 2147483647, #we want record only training part of epoch
                wait = 0,
                warmup = 1,
                active = 1,
                repeat = 1
            ),
            on_trace_ready=partial(
                trace_handler, 
                sort_by_keyword=self.cfg.base.profiling.sort_by_keyword if self.cfg.base.get("profiling", None) is not None else None, 
                phase="test",
                is_print=self.cfg.base.profiling.is_print if self.cfg.base.get("profiling", None) is not None else None
            )
        ) as prof:
            prof.step() #just to skip warmup and do not see warning

            self.model.eval()
            with torch.no_grad():    
                test_metrics_intersection = None
                test_metrics = self.__loop__("test",fold_ind, self.dataloaders['test'], self.cfg.base.device)
                if "test_intersection" in self.dataloaders:
                    test_metrics_intersection = self.__loop__("test",fold_ind, self.dataloaders['test_intersection'], self.cfg.base.device)
                if self.cfg.base.log.logging:
                    wandb.log({f"test/fold_{fold_ind}/{key}" : value for key, value in test_metrics.items()})
                    if "test_intersection" in self.dataloaders:
                        wandb.log({f"test_intersection/fold_{fold_ind}/{key}" : value for key, value in test_metrics_intersection.items()})

            return test_metrics, test_metrics_intersection   
        

class UnimodalSurvivalTrainer(Trainer):
    
    def __init__(self, splits: Dict[str,pd.DataFrame], cfg: DictConfig):
        
        super().__init__(splits, cfg)
        
        transforms = None
        if cfg.base.modalities[0]=="rna":
            if self.cfg.base.architecture=="MAE":
                 OmegaConf.set_struct(cfg, False) 
                 cfg.model["size"] = cfg.model.size if cfg.model.get("size", None) else math.ceil(len(self.preproc.get_column_order()) /cfg.model.patch_size)* cfg.model.patch_size
                 transforms = padded_transforms_with_scaling(self.preproc.get_scaling(), cfg.model.size)
                 print("transforms: ", transforms)
            elif self.cfg.base.architecture=="CNN":    
                transforms = base_transforms(self.preproc.get_scaling())
        elif self.cfg.base.modalities[0]=="dnam":
            transforms = padded_transforms_with_scaling(self.preproc.get_scaling(), cfg.model.get("size", None))
        elif self.cfg.base.modalities[0]=="clinical":
             transforms = base_scaling(self.preproc.get_scaling())       
        self.datasets = self.initialise_datasets(splits, self.cfg.base.modalities[0], self.preproc, transforms)

        self.dataloaders = {split: DataLoader(self.datasets[split],shuffle=True if split == "train" else False,
                                              batch_size=cfg.base.batch_size if split == "train" else 1,
                                              num_workers=self.cfg.base.get("num_workers", 0))
                            for split in splits.keys()}

        self.model =self.initialise_models().to(cfg.base.device)
        self.initialise_loss()

        print(self.model)
        print_vit_sizes(self.model)
        torch.cuda.reset_peak_memory_stats(device=cfg.base.device)
        print(f"gpu used {torch.cuda.max_memory_allocated(device=cfg.base.device)/1024/1024} Mb memory for model weights (without gradients and optimizer state)")
   
    def initialise_datasets(self, splits, modality, preproc, transforms=None):
        datasets ={}
        print("UnimodalSurvivalTrainer, initailise data")
        # Todo - подумать нужно ли тут разббить для каждого trainerа - свой initialise_dataset
        if modality == "rna":
            for split_name, dataset in splits.items():
                splits[split_name] = preproc.transform_labels(dataset)
                datasets[split_name] = RNASurvivalDataset(splits[split_name], self.cfg.data.rna.rna_dataset_path, 
                                                 transform = transforms, is_hazard_logits = True, column_order=preproc.get_column_order(), debug_mode = self.cfg.data.rna.get("debug_mode", False))

        elif modality == "mri":
                splits = {split_name: preproc.transform_labels(split) for split_name, split in splits.items()}
                for split_name, split in splits.items():
                    if self.cfg.data.get("embedding_name", None) is not None:
                        dataset = MRIEmbeddingDataset(split, return_mask=False, embedding_name=self.cfg.data.mri.embedding_name)
                        datasets[split_name] = SurvivalMRIDataset(split, dataset, is_hazard_logits=True)
                    else:
                        datasets[split_name] = MRISurvivalDataset(
                            split, 
                            self.cfg.data.mri.root_path,
                            self.cfg.data.mri.modalities, 
                            self.cfg.data.mri.sizes, 
                            transform = get_basic_tumor_transforms(self.cfg.data.mri.sizes) if self.cfg.data.mri.get("tensor_name", None) is None else None, 
                            return_mask=True, 
                            is_hazard_logits=True,
                            tensor_name=self.cfg.data.mri.get("tensor_name", None)
                        )
        elif modality == "dnam":
            for split_name, dataset in splits.items():
                splits[split_name] = preproc.transform_labels(dataset)
                datasets[split_name] = DNAmSurvivalDataset(splits[split_name], self.cfg.data.dnam.dnam_dataset_path, 
                                                 transform = transforms, is_hazard_logits = True, column_order=preproc.get_column_order())
        elif modality == "clinical":
            for split_name, dataset in splits.items():
                splits[split_name] = preproc.transform_labels(dataset)
                datasets[split_name] = ClinicalSurvivalDataset(splits[split_name], self.cfg.data.clinical.dataset_path, self.cfg.data.clinical.selected_columns,
                                                    transform = transforms, is_hazard_logits = True)
                    
        elif modality == "wsi":
            splits = {split_name: preproc.transform_labels(split) for split_name, split in splits.items()}
            for split_name, split in splits.items():
                    # Определяем значение параметра num в зависимости от типа раздела
                    is_train = True if split_name == "train" else False
                    if self.cfg.base.architecture=="CNN":
                        # Создаем датасет с нужными параметрами
                        dataset = WSIDataset(split, self.cfg.data.wsi.k, is_train=is_train, return_mask=True)
                        # Создаем SurvivalMRIDataset с нужными параметрами
                        datasets[split_name] = SurvivalWSIDataset(split, dataset, is_hazard_logits=True)
                    elif self.cfg.base.architecture=="MAE":
                        dataset = WSIDataset_patches(split, random_patch_selection = self.cfg.data.wsi.random_patch_selection, return_mask=True)
                        # Создаем SurvivalMRIDataset с нужными параметрами
                        datasets[split_name] = SurvivalWSIDataset(split, dataset, is_hazard_logits=True)
        else:
            raise NotImplementedError("Exist only for rna and mri. Initialising datasets for other modalities aren't declared")
        
        return datasets
 
    def initialise_models(self):
        # TODO check that is ok for multimodal
        if self.cfg.base.modalities[0]=="rna": 
                if self.cfg.base.architecture=="MAE":
                    return initialise_rna_mae_model(ViTMAEConfig(**OmegaConf.to_container(self.cfg.model)))
                elif self.cfg.base.architecture=="CNN":
                    return initialise_rna_model(self.cfg.model)
                else:
                    raise NotImplementedError("Exist only for rna. Initialising datasets for other modalities aren't declared")
        elif self.cfg.base.modalities[0]=="mri":
                if self.cfg.base.architecture=="Noise":
                    return MriNoiseSurvivalModel(ViTMAEConfig(**OmegaConf.to_container(self.cfg.model)))
                elif self.cfg.base.architecture=="MAE":
                    return MriMaeSurvivalModel(ViTMAEConfig(**OmegaConf.to_container(self.cfg.model)))
                elif self.cfg.base.architecture=="CNN":
                    return MRIEmbeddingEncoder(self.cfg.model.input_embedding_dim, self.cfg.model.dropout, self.cfg.base.n_intervals)
                else:
                    raise NotImplementedError("Exist only Noise, MAE and CNN architectures for mri modality")
        elif self.cfg.base.modalities[0]=="dnam":
                if self.cfg.base.architecture=="MAE":
                    return initialise_dnam_mae_model(ViTMAEConfig(**OmegaConf.to_container(self.cfg.model)))
                elif self.cfg.base.architecture=="CNN":
                    return initialise_dnam_model(self.cfg.model)
                else:
                    raise NotImplementedError("Exist only for rna. Initialising datasets for other modalities aren't declared")
        elif self.cfg.base.modalities[0]=="wsi":
                if self.cfg.base.architecture=="MAE":
                    return WsiMaeSurvivalModel(ViTMAEConfig(**OmegaConf.to_container(self.cfg.model)))
                elif self.cfg.base.architecture=="CNN":
                     return WSIEncoder(embedding_dim=self.cfg.model.input_embedding_dim, depth=self.cfg.model.depth,
                                      heads=self.cfg.model.heads, dim=self.cfg.model.dim, pool=self.cfg.model.pool,
                                      dim_head=self.cfg.model.dim_head, mlp_dim=self.cfg.model.mlp_dim, dropout=self.cfg.model.dropout,
                                      emb_dropout=self.cfg.model.emb_dropout, n_outputs=self.cfg.model.n_outputs)
                else:
                    raise NotImplementedError("Exist only MAE and CNN architectures for mri modality")    
        else:
                raise NotImplementedError("Exist only for rna and mri. Initialising models for other modalities aren't declared")   

            
    def initialise_loss(self):    
        self.criterion = NLLLogistiHazardLoss()
        self.optimizer = AdamW(self.model.parameters(), **self.cfg.base.optimizer.params)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.base.n_epochs,**self.cfg.base.scheduler.params)
        
    def __loop__(self,split, fold_ind, dataloader, device):
        total_task_loss =0
        num_samples = 0
        preds,times,events = [], [], []

        for batch in dataloader:
            
            data, mask,  time, event = batch 
            data = {modality :value.to(device) for modality, value in data.items()} if isinstance(data, dict) else data.to(device)

            outputs =self.model(data, masks = mask)
    
            loss = self.criterion(outputs, time.to(device), event.to(device))
                
            # Backpropagation
            if split=="train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            preds.append(outputs)
            times.append(time)
            events.append(event)
            total_task_loss+=loss*len(batch)
            num_samples+=len(batch)
            
        metrics = {"task_loss": total_task_loss.cpu().detach().numpy() / num_samples}
        if split!="train":
            preproc = self.preproc[next(iter(self.preproc))] if isinstance(self.preproc, dict) else self.preproc
            metrics.update(compute_survival_metrics( preds, torch.cat(times, dim=0), torch.cat(events, dim=0), cuts=preproc.get_hazard_cuts()))
        else:
            self.scheduler.step()
            
        return  metrics 

        
class UnimodalMAETrainer(Trainer):
    
    def __init__(self, splits: Dict[str,pd.DataFrame], cfg: DictConfig):
        super().__init__(splits, cfg)
        transforms = None
        if self.cfg.base.modalities[0]=="rna":
            OmegaConf.set_struct(cfg, False) 
            cfg.model["size"] = cfg.model.size if cfg.model.get("size", None) else math.ceil(len(self.preproc.get_column_order()) /cfg.model.patch_size)* cfg.model.patch_size
            transforms = padded_transforms_with_scaling(self.preproc.get_scaling(), cfg.model.size)
            print("transforms: ", transforms)
        elif self.cfg.base.modalities[0]=="dnam":
            transforms = padded_transforms_with_scaling(self.preproc.get_scaling(), cfg.model.get("size", None))
        elif self.cfg.base.modalities[0]=="mri":
            transforms = None
        elif self.cfg.base.modalities[0]=="wsi":
            transforms = None
        else:
            raise NotImplementedError("Exist only for rna and mri. Initialising datasets for other modalities aren't declared")
        self.datasets = self.initialise_datasets(splits, self.cfg.base.modalities[0], self.preproc, transforms)
        self.dataloaders = {split: DataLoader(self.datasets[split],shuffle=True if split == "train" else False, batch_size=cfg.base.batch_size  
                                              if split == "train" else 1, num_workers=self.cfg.base.get("num_workers", 0))
                            for split in splits.keys()}

        self.model =self.initialise_models().to(cfg.base.device)
        print(self.model)
        print_vit_sizes(self.model)
        torch.cuda.reset_peak_memory_stats(device=cfg.base.device)
        print(f"gpu used {torch.cuda.max_memory_allocated(device=cfg.base.device)/1024/1024} Mb memory")
        
        self.initialise_loss()

    
    def initialise_loss(self):
        self.optimizer = AdamW(self.model.parameters(), **self.cfg.base.optimizer.params)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.base.n_epochs,**self.cfg.base.scheduler.params)
        
    def initialise_models(self):
        if self.cfg.base.modalities[0]=="rna":
            return  RnaMAEForPreTraining(ViTMAEConfig(**OmegaConf.to_container(self.cfg.model)))
        elif self.cfg.base.modalities[0]=="mri":
            return MriMAEForPreTraining(ViTMAEConfig(**OmegaConf.to_container(self.cfg.model)))
        elif self.cfg.base.modalities[0]=="dnam":
            return DNAmMAEForPreTraining(ViTMAEConfig(**OmegaConf.to_container(self.cfg.model)))
        elif self.cfg.base.modalities[0]=="wsi":
            return WsiMAEForPreTraining(ViTMAEConfig(**OmegaConf.to_container(self.cfg.model)))     
        else:
            raise NotImplementedError("Exist only for rna and mri. Initialising models for other modalities aren't declared") 
        
    def initialise_datasets(self, splits, modality, preproc=None, transforms=None):
        datasets ={}
        # Todo - подумать нужно ли тут разббить для каждого trainerа - свой initialise_dataset
        if modality == "rna":
            for split_name, dataset in splits.items():
                splits[split_name] = preproc.transform_labels(dataset)
                datasets[split_name] = RNADataset(splits[split_name], self.cfg.data.rna.rna_dataset_path, 
                                                 transform = transforms, is_hazard_logits = True, column_order=preproc.get_column_order())

        elif modality == "mri":
            for split_name, split in splits.items():
                print(self.cfg.data)
                datasets[split_name] = MRIDataset(
                    split, 
                    self.cfg.data.mri.root_path,
                    self.cfg.data.mri.modalities, 
                    self.cfg.data.mri.sizes, 
                    transform = get_basic_tumor_transforms(self.cfg.data.mri.sizes) if self.cfg.data.mri.get("tensor_name", None) is None else None,
                    return_mask=True,
                    tensor_name=self.cfg.data.mri.get("tensor_name", None)  
                )
        elif modality == "dnam":
            for split_name, dataset in splits.items():
                splits[split_name] = preproc.transform_labels(dataset)
                datasets[split_name] = DNAmDataset(splits[split_name], self.cfg.data.dnam.dnam_dataset_path, 
                                                 transform = transforms, is_hazard_logits = True, column_order=preproc.get_column_order())
        elif modality == "wsi":
            for split_name, split in splits.items():
                    # Определяем значение параметра num в зависимости от типа раздела                  
                    # Создаем датасет с нужными параметрами
                    # if len(self.cfg.base.modalities)>1:
                    #     is_train = True if split_name == "train" else False
                    #     datasets[split_name] = WSIDataset(split,self.cfg.data.wsi.k, is_train=is_train,return_mask=True)
                    # else:
                    datasets[split_name] = WSIDataset_patches(split, random_patch_selection = self.cfg.data.wsi.random_patch_selection, return_mask=True)
        else:
            raise NotImplementedError("Exist only for rna and mri. Initialising datasets for other modalities aren't declared")
        
        return datasets
        
    def __loop__(self,split, fold_ind, dataloader, device):
        total_loss =0
        num_samples = 0

        for batch in tqdm(dataloader):
            if isinstance(batch, tuple) or isinstance(batch, list): data, mask = batch 
            elif isinstance(batch, dict): data = batch["image"]
            else: data = batch

            outputs =self.model(data.to(device))
            
            if split=="train":
                self.optimizer.zero_grad()
                outputs.loss.backward()
                self.optimizer.step()
            
            total_loss+=outputs.loss*len(batch) #outputs.loss - mean loss across batch
            num_samples+=len(batch)
        
        metrics = {"mse_loss": total_loss.cpu().detach().numpy() / num_samples}
        
        if split=="train":
            self.scheduler.step()
            
        return  metrics

class UnimodalNoiseTrainer(UnimodalMAETrainer):
    
    def __init__(self, splits: Dict[str,pd.DataFrame], cfg: DictConfig):
        super().__init__(splits, cfg)

    def initialise_models(self):
        if self.cfg.base.modalities[0]=="mri":
            return MriNoiseForPreTraining(ViTMAEConfig(**OmegaConf.to_container(self.cfg.model)))  
        else:
            raise NotImplementedError("Exist only for mri. Initialising models for other modalities aren't declared") 