import numpy as np
from typing import Dict, List, Tuple, Union
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from src.unimodal.rna.dataset import RNADataset
from src.unimodal.mri.datasets import SurvivalMRIEmbeddingDataset
from src.unimodal.rna.preprocessor import RNAPreprocessor
from src.preprocessor import BaseUnimodalPreprocessor
from src.unimodal.rna.transforms import base_transforms, padded_transforms
from torch.utils.data import Dataset, DataLoader
from pycox.models.loss import NLLLogistiHazardLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR   
from src.unimodal.rna.encoder import initialise_rna_model
from src.unimodal.mri.models import MRIEmbeddingEncoder
from src.unimodal.rna.mae import initialise_rna_mae_model
import torch
from ..evaluation import compute_survival_metrics
import wandb
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from src.unimodal.rna.mae import RnaMAEForPreTraining
from src.utils import check_dir_exists
from tqdm.auto import tqdm

from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
from src.unimodal.rna.transforms import UpperQuartileNormalizer

class Trainer(object):
    def __init__(self, splits: Dict[str,pd.DataFrame], cfg: DictConfig):
        self.cfg =cfg
        self.preproc = self.initialise_preprocessing(splits, self.cfg.base.modalities[0])
 
    def initialise_preprocessing(self, splits, modality):
        
        if modality=="rna":
            
            if self.cfg.data.rna.scaling_method in ["QuantileTransformer", "StandardScaler", "MinMaxScaler"]:
                scaling_method = getattr(__import__('sklearn.preprocessing', fromlist=[self.cfg.data.rna.scaling_method]), self.cfg.data.rna.scaling_method)
            elif self.cfg.data.rna.scaling_method=="UpperQuartileNormalizer":
                 scaling_method = UpperQuartileNormalizer 
            print("Scaling method: ", scaling_method)
            preproc = RNAPreprocessor(splits["train"], self.cfg.base.rna_dataset_path, self.cfg.base.n_intervals, scaling_method, 
                                          self.cfg.data.rna.scaling_params, self.cfg.data.rna.var_threshold,
                                          self.cfg.data.rna.is_cluster_genes , self.cfg.data.rna.clustering_threshold)
            preproc.fit()
            return preproc

        elif modality == "mri":
            preproc = BaseUnimodalPreprocessor(splits["train"], self.cfg.base.n_intervals)
            preproc.fit()
            return preproc

        else:
            raise NotImplementedError("Exist only for rna and mri. Initialising preprocessing for other modalities aren't declared")    
                
    def initialise_datasets(self, splits, modality, transforms=None):
        datasets ={}

        if modality == "rna":
            for split_name, dataset in splits.items():
                splits[split_name] = self.preproc.transform_labels(dataset)
                datasets[split_name] = RNADataset(splits[split_name], self.cfg.base.rna_dataset_path, 
                                                 transform = transforms, is_hazard_logits = True, column_order=self.preproc.get_column_order())

        elif modality == "mri":
            splits = {split_name: self.preproc.transform_labels(split) for split_name, split in splits.items()}
            datasets = {
                split_name: SurvivalMRIEmbeddingDataset(
                    split,
                    is_hazard_logits = True,
                    embedding_name=self.cfg.data.embedding_name
                ) 
                for split_name, split in splits.items()
            }

        else:
            raise NotImplementedError("Exist only for rna and mri. Initialising datasets for other modalities aren't declared")
        
        return datasets
   
    
    def train(self, fold_ind : int):
        # best_loss = np.infty
        # best_epoch = -1

        for epoch in tqdm(range(self.cfg.base.n_epochs)):
            self.model.train()
            
            train_metrics = self.__loop__("train",fold_ind, self.dataloaders['train'], self.cfg.base.device)
            train_metrics.update({"epoch": epoch})
            if self.cfg.base.log.logging:
                wandb.log({f"train/fold_{fold_ind}/{key}" : value for key, value in train_metrics.items()})
            
            self.model.eval()
            with torch.no_grad():    
                val_metrics = self.__loop__("val",fold_ind, self.dataloaders['val'], self.cfg.base.device)
            val_metrics.update({"epoch": epoch})    
            if self.cfg.base.log.logging:
                wandb.log({f"val/fold_{fold_ind}/{key}" : value for key, value in val_metrics.items()})
                
        # if val_metrics[self.loss_key] < best_loss:
            # best_loss = val_metrics[self.loss_key]
            # best_epoch = epoch
        check_dir_exists(self.cfg.base.save_path)
        torch.save(self.model.state_dict(), self.cfg.base.save_path)

        # print(f"Best loss: {best_loss} at epoch {best_epoch}")
        return val_metrics
    
    def evaluate(self, fold_ind : int):
        self.model.eval()
        with torch.no_grad():    
            test_metrics = self.__loop__("test",fold_ind, self.dataloaders['test'], self.cfg.base.device)
        if self.cfg.base.log.logging:
            wandb.log({f"test/fold_{fold_ind}/{key}" : value for key, value in test_metrics.items()})    
        return test_metrics    
        

class UnimodalSurvivalTrainer(Trainer):
    
    def __init__(self, splits: Dict[str,pd.DataFrame], cfg: DictConfig):
        
        super().__init__(splits, cfg)
        
        transforms = None
        if cfg.base.modalities[0]=="rna":
            if self.cfg.base.architecture=="MAE":
                transforms = padded_transforms(self.preproc.get_scaling(), cfg.model.rna_size)
            elif self.cfg.base.architecture=="CNN":    
                transforms = base_transforms(self.preproc.get_scaling())
                
        self.datasets = self.initialise_datasets(splits, self.cfg.base.modalities[0], transforms)

        self.dataloaders = {"train" : DataLoader(self.datasets["train"],shuffle=True, batch_size =cfg.base.batch_size),
                            "val" : DataLoader(self.datasets["val"],shuffle=False, batch_size = 1),
                            "test" : DataLoader(self.datasets["test"],shuffle=False, batch_size =1)
                            }

        self.model =self.initialise_models().to(cfg.base.device)
        self.initialise_loss()
        self.loss_key = "task_loss"
        print(self.model)
    
    def initialise_models(self):

        if self.cfg.base.modalities[0]=="rna": 
                if self.cfg.base.architecture=="MAE":
                    return initialise_rna_mae_model(ViTMAEConfig(**OmegaConf.to_container(self.cfg.model)))
                elif self.cfg.base.architecture=="CNN":
                    return initialise_rna_model(self.cfg.model)
                else:
                    raise NotImplementedError("Exist only for rna. Initialising datasets for other modalities aren't declared")
        elif self.cfg.base.modalities[0]=="mri":
            
                return MRIEmbeddingEncoder(self.cfg.model.input_embedding_dim, self.cfg.model.dropout, self.cfg.base.n_intervals)

        else:
                raise NotImplementedError("Exist only for rna and mri. Initialising models for other modalities aren't declared")   

            
    def initialise_loss(self):    
        self.criterion = NLLLogistiHazardLoss()
        self.optimizer = AdamW(self.model.parameters(), **self.cfg.base.optimizer.params)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.base.n_epochs,**self.cfg.base.scheduler.params)
        
    def __loop__(self,split, fold_ind, dataloader, device):
        total_task_loss =0
        preds,times,events = [], [], []
        for batch in dataloader:
  
                  
            data, time, event = batch  
            outputs =self.model(data.to(device))
    
            loss = self.criterion(outputs, time.to(device), event.to(device))
                
            # Backpropagation
            if split=="train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            preds.append(outputs)
            times.append(time)
            events.append(event)
            total_task_loss+=loss
            
        metrics = {self.loss_key: total_task_loss.cpu().detach().numpy() / len(dataloader.dataset)}
        if split!="train":
            metrics.update(compute_survival_metrics( preds, torch.cat(times, dim=0), torch.cat(events, dim=0), cuts=self.preproc.get_hazard_cuts()))
        else:
            self.scheduler.step()
            
        return  metrics  
                    
        
        
class UnimodalMAETrainer(Trainer):
    
    def __init__(self, splits: Dict[str,pd.DataFrame], cfg: DictConfig):
        super().__init__(splits, cfg)
        transforms = padded_transforms(self.preproc.get_scaling(), cfg.model.rna_size)
        self.datasets = self.initialise_datasets(splits, self.cfg.base.modalities[0], transforms)
        self.dataloaders = {"train" : DataLoader(self.datasets["train"],shuffle=True, batch_size =cfg.base.batch_size),
                            "val" : DataLoader(self.datasets["val"],shuffle=False, batch_size = 1),
                            "test" : DataLoader(self.datasets["test"],shuffle=False, batch_size =1)
                            }
        self.model =self.initialise_models().to(cfg.base.device)
        print(self.model)
        self.initialise_loss()
        self.loss_key = "mse_loss"
    
    def initialise_loss(self):
        self.optimizer = AdamW(self.model.parameters(), **self.cfg.base.optimizer.params)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.base.n_epochs,**self.cfg.base.scheduler.params)
        
    def initialise_models(self):
        if self.cfg.base.modalities[0]=="rna":
                return RnaMAEForPreTraining(ViTMAEConfig(**OmegaConf.to_container(self.cfg.model)))
        else:
            raise NotImplementedError("Exist only for rna. Initialising datasets for other modalities aren't declared") 
        
    def __loop__(self,split, fold_ind, dataloader, device):
        total_loss =0
        
        for batch in dataloader:
            data, time, event = batch  
            outputs =self.model(data.to(device))
            
            if split=="train":
                self.optimizer.zero_grad()
                outputs.loss.backward()
                self.optimizer.step()
            
            total_loss+=outputs.loss
        
        metrics = {self.loss_key: total_loss.cpu().detach().numpy() / len(dataloader.dataset)}
        
        if split=="train":
            self.scheduler.step()
            
        return  metrics