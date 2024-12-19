from typing import Dict, List, Tuple, Union
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from src.unimodal.rna.dataset import RNADataset
from src.unimodal.mri import MRIEmbeddingDataset
from src.unimodal.rna.preprocessor import RNAPreprocessor
from src.unimodal.rna.transforms import base_transforms, padded_transforms
from torch.utils.data import Dataset, DataLoader
from pycox.models.loss import NLLLogistiHazardLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR   
from src.unimodal.rna.encoder import initialise_rna_model
import torch
from ..evaluation import compute_survival_metrics
import wandb
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from src.unimodal.rna.mae import RnaMAEForPreTraining


class Trainer(object):
    def __init__(self, splits: Dict[str,pd.DataFrame], cfg: DictConfig):
        
        self.cfg =cfg
        self.initialise_preprocessing(splits)
 
    def initialise_preprocessing(self, splits):
         if self.cfg.base.modalities[0]=="rna":
            self.preproc =RNAPreprocessor(splits["train"], self.cfg.base.rna_dataset_path, self.cfg.base.n_intervals, self.cfg.data.rna.is_cluster_genes , self.cfg.data.rna.clustering_threshold)
            self.preproc.fit()
         else:
            raise NotImplementedError("Exist only for rna. Initialising datasets for other modalities aren't declared")    
                
    def initialise_datasets(self, splits, transforms):
        datasets ={}
        for split_name, dataset in splits.items():
            splits[split_name] = self.preproc.transform_labels(dataset)
            datasets[split_name] = RNADataset(splits[split_name], self.cfg.base.rna_dataset_path, 
                                                 transform = transforms, is_hazard_logits = True, column_order=self.preproc.get_column_order())
        return datasets
   
    
    def train(self, fold_ind : int):
    
        for epoch in range(self.cfg.base.n_epochs):
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
        torch.save(self.model.state_dict(), self.cfg.base.save_path)
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
        transforms = base_transforms(self.preproc.get_scaling())
        self.datasets = self.initialise_datasets(splits, transforms)
        self.dataloaders = {"train" : DataLoader(self.datasets["train"],shuffle=True, batch_size =cfg.base.batch_size),
                            "val" : DataLoader(self.datasets["val"],shuffle=False, batch_size = 1),
                            "test" : DataLoader(self.datasets["test"],shuffle=False, batch_size =1)
                            }
        self.model =self.initialise_models().to(cfg.base.device)
        self.initialise_loss()
        print(self.model)
    
    
<<<<<<< Updated upstream
=======
    
    def initialise_datasets(self, splits):
        datasets ={}
        if self.cfg.base.modalities[0]=="rna":
            self.preproc =RNAPreprocessor(splits["train"], self.cfg.base.rna_dataset_path, self.cfg.base.n_intervals, self.cfg.data.rna.is_cluster_genes , self.cfg.data.rna.clustering_threshold)
            self.preproc.fit()
            for split_name, dataset in splits.items():
                splits[split_name] =self.preproc.transform_labels(dataset)
                transforms = base_transforms(self.preproc.get_scaling())
                datasets[split_name] =RNADataset(splits[split_name], self.cfg.base.rna_dataset_path, 
                                                 transform = transforms, is_hazard_logits = True, column_order=self.preproc.get_column_order())
            return datasets

        elif self.cfg.base.modalities[0]=="mri":
            return {split_name: MRIEmbeddingDataset(split, return_mask=False) for split_name, split in splits.items()}

        else:
            raise NotImplementedError("Exist only for RNA and MRI. Initialising datasets for other modalities aren't declared")
    
>>>>>>> Stashed changes
    def initialise_models(self):
           if self.cfg.base.modalities[0]=="rna": 
                    return initialise_rna_model(self.cfg.model)
           else:
               raise NotImplementedError("Exist only for rna. Initialising datasets for other modalities aren't declared")   
            
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
            
        metrics = {"task_loss": total_task_loss.cpu().detach().numpy() / len(dataloader.dataset)}
        if split!="train":
            metrics.update(compute_survival_metrics( preds, torch.cat(times, dim=0), torch.cat(events, dim=0), cuts=self.preproc.get_hazard_cuts()))
        else:
            self.scheduler.step()
            
        return  metrics  
                    
        
        
class UnimodalMAETrainer(Trainer):
    
    def __init__(self, splits: Dict[str,pd.DataFrame], cfg: DictConfig):
        super().__init__(splits, cfg)
        transforms = padded_transforms(self.preproc.get_scaling(), cfg.model.rna_size)
        self.datasets = self.initialise_datasets(splits, transforms)
        self.dataloaders = {"train" : DataLoader(self.datasets["train"],shuffle=True, batch_size =cfg.base.batch_size),
                            "val" : DataLoader(self.datasets["val"],shuffle=False, batch_size = 1),
                            "test" : DataLoader(self.datasets["test"],shuffle=False, batch_size =1)
                            }
        self.model =self.initialise_models().to(cfg.base.device)
        print(self.model)
        self.initialise_loss()
    
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
        
        metrics = {"mse_loss": total_loss.cpu().detach().numpy() / len(dataloader.dataset)}
        
        if split=="train":
            self.scheduler.step()
            
        return  metrics