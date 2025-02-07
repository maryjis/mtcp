from collections import defaultdict
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from typing import Dict
from src.multimodal.datasets import MultimodalDataset, MultimodalSurvivalDataset
from torch.utils.data import DataLoader
from pycox.models.loss import NLLLogistiHazardLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from src.unimodal.rna.transforms import base_transforms, padded_transforms
from src.unimodal.dna.transforms import padded_transforms_simple
from src.unimodal.clinical.transforms import base_scaling
from src.unimodal.trainer import Trainer, UnimodalSurvivalTrainer, UnimodalMAETrainer
from src.multimodal.models import MultiMaeForPretraining, MultiMaeForSurvival
import torch

class MultiModalTrainer(Trainer):
    def __init__(self, splits: Dict[str,pd.DataFrame], cfg: DictConfig):
        self.cfg =cfg
        self.preproc = self.initialise_preprocessing(splits, self.cfg.base.modalities)
        print(self.preproc)
 
    def initialise_preprocessing(self, splits, modalities):
        preproc_dict = {}
        for modality in modalities:
            preproc_dict[modality] =super().initialise_preprocessing(splits, modality)       
        return preproc_dict

# def collate_fn(batch):
#     items, masks = defaultdict(list), defaultdict(list)
#     for elem in batch:
#         for modality in elem[0].keys():
#             items[modality].append(elem[0][modality])
#             masks[modality].append(elem[1][modality])
#     for modality in items.keys():
#         print(modality)
#         print(masks[modality])
#         items[modality] = torch.stack(items[modality])
#         masks[modality] = torch.tensor(masks[modality])
#     return items, masks

class MultiModalMAETrainer(MultiModalTrainer, UnimodalMAETrainer):
    
    def __init__(self, splits: Dict[str,pd.DataFrame], cfg: DictConfig):
        super().__init__(splits, cfg)
        # TODO MRI - done preprocess! 
        transforms = {"rna": padded_transforms(self.preproc["rna"].get_scaling(), cfg.model.rna_model.size), 
                      "dnam" : padded_transforms_simple(cfg.model.dnam_model.get("size", None)) if cfg.model.get("dnam_model", None) else None, "mri" : None, "wsi" : None }
        self.datasets = self.initialise_datasets(splits, self.cfg.base.modalities, self.preproc, transforms)
        self.dataloaders = {split: DataLoader(self.datasets[split],shuffle=True if split == "train" else False, batch_size=cfg.base.batch_size 
                                              if split == "train" else 1)
                            for split in splits.keys()}
        self.model =self.initialise_models().to(cfg.base.device)
        print(self.model)
        self.initialise_loss()
    
    def initialise_loss(self):
        self.optimizer = AdamW(self.model.parameters(), **self.cfg.base.optimizer.params)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.base.n_epochs,**self.cfg.base.scheduler.params)
        
    def initialise_models(self):
        return MultiMaeForPretraining(ViTMAEConfig(**OmegaConf.to_container(self.cfg.model)))
 
        
    def __loop__(self,split, fold_ind, dataloader, device):
        total_loss =0
        modality_losses = {}
        for batch in dataloader:
            data, masks = batch 
            data = {modality :value.to(device) for modality, value in data.items()} 
            outputs =self.model(data,masks)
            
            if split=="train":
                self.optimizer.zero_grad()
                outputs.loss[0].backward()
                self.optimizer.step()
            
            total_loss+=outputs.loss[0]
            for modality, _ in data.items():
                if modality not in modality_losses:
                    modality_losses[modality] = 0
                modality_losses[modality] += outputs.loss[1][modality]    
        metrics = {"mse_loss": total_loss.cpu().detach().numpy() / len(dataloader.dataset)}
        modalities = self.cfg.base.modalities
        for modality in modalities:
            metrics[f"mse_{modality}_loss"] = modality_losses[modality].cpu().detach().numpy() / len(dataloader.dataset)
        if split=="train":
            self.scheduler.step()
            
        return  metrics
    
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
            concat_dataset[split] = MultimodalDataset(splits[split], self.cfg.base.data_path, modalities_data,
                                                 transform = transforms, is_hazard_logits = True)
        return concat_dataset
    
    
class MultiModalSurvivalTrainer(MultiModalTrainer, UnimodalSurvivalTrainer):
    def __init__(self, splits: Dict[str,pd.DataFrame], cfg: DictConfig):
        super().__init__(splits, cfg)
        ## TODO MRI add transforms!!
        transforms = {"rna": padded_transforms(self.preproc["rna"].get_scaling(), cfg.model.rna_model.size), 
                      "dnam" : padded_transforms_simple(cfg.model.dnam_model.get("size", None)) if cfg.model.get("dnam_model", None) else None,
                      "mri" : None, "wsi" : None,
                      "clinical" : base_scaling(self.preproc["clinical"].get_scaling() if "clinical" in self.preproc.keys() else None)}
        self.datasets = self.initialise_datasets(splits, self.cfg.base.modalities, self.preproc, transforms)
        self.dataloaders = {split: DataLoader(self.datasets[split],shuffle=True if split == "train" else False, batch_size=cfg.base.batch_size 
                                              if split == "train" else 1)
                            for split in splits.keys()}
        self.model = self.initialise_models().to(cfg.base.device)
        self.initialise_loss()
        print(self.model)
    
    def initialise_loss(self):    
        self.criterion = NLLLogistiHazardLoss()
        self.optimizer = AdamW(self.model.parameters(), **self.cfg.base.optimizer.params)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.base.n_epochs,**self.cfg.base.scheduler.params)
        
    def initialise_models(self):
        return MultiMaeForSurvival(ViTMAEConfig(**OmegaConf.to_container(self.cfg.model)))
    
    def initialise_datasets(self, splits, modalities, preprocs, transforms=None):
        datasets = defaultdict(list)
        for modality in modalities:
            transform = transforms[modality] if transforms is not None else None
            print(modality, transform)
            mdata = super().initialise_datasets(splits,modality,preprocs[modality],transform)
            for key, value in mdata.items():
                datasets[key].append((modality,value))
        # Concat datasets
        concat_dataset = {} 
        for split, modalities_data in datasets.items():
            concat_dataset[split] = MultimodalSurvivalDataset(splits[split], self.cfg.base.data_path, modalities_data,
                                                 transform = transforms, is_hazard_logits = True)
        return concat_dataset
    


 
