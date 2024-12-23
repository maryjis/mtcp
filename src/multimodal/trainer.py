from src.unimodal import Trainer
from collections import defaultdict
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Tuple, Union
from src.multimodal.datasets import MultimodalDataset
from torch.utils.data import Dataset, DataLoader
from pycox.models.loss import NLLLogistiHazardLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from src.unimodal.rna.transforms import base_transforms, padded_transforms

class MultiModalTrainer(Trainer):
    def __init__(self, splits: Dict[str,pd.DataFrame], cfg: DictConfig):
        self.cfg =cfg
        self.preproc = self.initialise_preprocessing(splits, self.cfg.base.modalities)
 
    def initialise_preprocessing(self, splits, modalities):
        preproc_dict = {}
        for modality in modalities:
            preproc_dict[modality] =super().initialise_preprocessing(splits, modality)       
        return preproc_dict

    def initialise_datasets(self, splits, modalities, preprocs, transforms=None):
        datasets = defaultdict(list)
        for modality in modalities:
            transform = transforms[modality] if transforms is not None else None
            mdata = super().initialise_datasets(splits,modality,preprocs[modality],transform)
            for key, value in mdata.items():
                datasets[key].append((modality,value))
        # Concat datasets
        concat_dataset = {} 
        for split, modalities_data in datasets.items():
            concat_dataset[split] = MultimodalDataset(splits[split], self.cfg.base.dataset_path, 
                                                 transform = transforms, is_hazard_logits = True)
        return concat_dataset

class MultiModalMAETrainer(Trainer):
    
    def __init__(self, splits: Dict[str,pd.DataFrame], cfg: DictConfig):
        super().__init__(splits, cfg)
        transforms = {"rna": padded_transforms(self.preproc.get_scaling(), cfg.model.rna_size), "mri" : None, "wsi" : None }
        self.datasets = self.initialise_datasets(splits, self.cfg.base.modalities, self.preproc, transforms)
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