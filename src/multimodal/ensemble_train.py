import torch
import torch.nn as nn
import torch.optim as optim
from pycox.models.loss import NLLLogistiHazardLoss
import torchtuples as tt
from src.multimodal.trainer import MultiModalTrainer
from src.unimodal.trainer import UnimodalSurvivalTrainer
from omegaconf import DictConfig
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict
from collections import OrderedDict, defaultdict
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from src.unimodal.rna.transforms import padded_transforms_with_scaling
from src.unimodal.dna.transforms import padded_transforms_scaling
from src.unimodal.cnv.transforms import padded_transforms_cnv_scaling
from src.unimodal.clinical.transforms import base_scaling
from src.multimodal.datasets import MultimodalDataset, MultimodalSurvivalDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from ..evaluation import compute_survival_metrics

class MultimodalBoostingSurvivalTrainer(MultiModalTrainer, UnimodalSurvivalTrainer):
    
    def __init__(self, splits: Dict[str,pd.DataFrame], cfg: DictConfig):
        super().__init__(splits, cfg)
                ## TODO MRI add transforms!!
        cfg.model.rna_model.size = cfg.model.rna_model.size if cfg.model.rna_model.get("size", None) else math.ceil(len(self.preproc["rna"].get_column_order()) /cfg.model.rna_model.patch_size)* cfg.model.rna_model.patch_size
        transforms = {"rna": padded_transforms_with_scaling(self.preproc["rna"].get_scaling(), cfg.model.rna_model.size),
                      "dnam" : padded_transforms_scaling(self.preproc["dnam"].get_scaling(), cfg.model.dnam_model.get("size", None)) if cfg.model.get("dnam_model", None) else None,
                      "mri" : None, "wsi" : None,
                      "cnv" : padded_transforms_cnv_scaling(self.preproc["cnv"].get_scaling(), cfg.model.cnv_model.get("size", None)) if "cnv" in self.preproc.keys() else None,
                      "clinical" : base_scaling(self.preproc["clinical"].get_scaling() if "clinical" in self.preproc.keys() else None)}
        self.datasets = self.initialise_datasets(splits, self.cfg.base.modalities, self.preproc, transforms)
        self.dataloaders = {split: DataLoader(self.datasets[split],shuffle=True if split == "train" else False, batch_size=cfg.base.batch_size 
                                              if split == "train" else 1, drop_last=True if split == "train" else False)
                            for split in splits.keys()}
        self.modalities =self.cfg.base.modalities
        self.lambdas = self.cfg.base.lambdas
        self.models = self.initialise_set_of_models()
        self.initialise_loss()
        print(self.models)
        
    def initialise_set_of_models(self):
        models =  []
        for modality in self.modalities:
            models.append((modality, self.initialise_models(modality, self.cfg.model[f"{modality}_model"]).to(self.cfg.base.device)))
        return models
    
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
    
    def initialise_loss(self):
        self.criterions = [NLLLogistiHazardLoss(reduction='none')]
        self.optimizers = [AdamW(self.models[0][1].parameters(), **self.cfg.base.optimizers[0].params)]
        self.schedulers = [CosineAnnealingLR(self.optimizers[0], T_max=self.cfg.base.n_epochs,**self.cfg.base.schedulers[0].params)]
        for i in range(1,len(self.modalities)):
            self.criterions.append(nn.MSELoss())
            self.optimizers.append(AdamW(self.models[i][1].parameters(), **self.cfg.base.optimizers[i].params))
            self.schedulers.append(CosineAnnealingLR(self.optimizers[i], T_max=self.cfg.base.n_epochs_boosting,**self.cfg.base.schedulers[i].params))
      
    def _forward_all(self, data, mask , max_i = None):
        """
        xs: list of modality inputs, each tensor of shape (batch, feat_dim)
        Returns: list of (batch, K) tensors of logits from each model
        """
        outputs = []
        if not max_i:
            max_i =len(self.models)
        else:
            max_i=max_i+1
        for i in range(max_i):
            modality_name = self.models[i][0]
            output =self.models[i][1](data[modality_name], masks = mask[modality_name])
            outputs.append(output)
            
        return outputs

    def _compose_logits(self, logits_list, max_i=None):
        """
        Sums up logits from models[0] to models[max_i] with weights
        """
        if not max_i:
            max_i =len(self.models)
        else:
            max_i=max_i+1
        eta = 0
        for i in range(max_i):
            eta += self.lambdas[i] * logits_list[i]
        return eta
    
    def predict(self, batch , max_i = None):
        logits = self._forward_all( batch, max_i)
        return self._compose_logits(logits, max_i)
    
    def __loop__(self,split, fold_ind, dataloader, device):
        total_task_loss =0
        num_samples = 0
        preds,times,events = [], [], []

        for batch in dataloader:
            
            data, mask,  time, event = batch 
            data = {modality :value.to(device) for modality, value in data.items()}
            modality_name = self.models[0][0]

            outputs =self.models[0][1](data[modality_name], masks = mask[modality_name])

            loss = self.criterions[0](outputs, time.to(device), event.to(dtype=torch.float32,device=device))
            loss = loss.mean() 
            # Backpropagation
            if split=="train":
                self.optimizers[0].zero_grad()
                loss.backward()
                self.optimizers[0].step()
            preds.append(outputs)
            times.append(time)
            events.append(event)
            total_task_loss+=loss*len(batch)
            num_samples+=len(batch)
            
        metrics = {"task_loss_first": total_task_loss.cpu().detach().numpy() / num_samples}
        if split!="train":
            preproc = self.preproc[next(iter(self.preproc))] if isinstance(self.preproc, dict) else self.preproc
            metrics.update(compute_survival_metrics( preds, torch.cat(times, dim=0), torch.cat(events, dim=0), cuts=preproc.get_hazard_cuts()))
        else:
            self.schedulers[0].step()
            
        return  metrics 
    
    def __boosting_step__(self,split,fold_ind, dataloader, i, device):
        total_task_loss =0
        num_samples = 0
        for batch in dataloader:
            data, mask,  time, event = batch
            data = {modality :value.to(device) for modality, value in data.items()}
            logits = self._forward_all(data, mask,  max_i=i)
            eta_prev = self._compose_logits(logits, max_i=i)  # η = sum of prev models

            losses = self.criterions[0](eta_prev, time.to(device), event.to(dtype=torch.float32,device=device))  # (N,)
            # 2) Вычисляем pseudo‐остатки g_i = ∂Σℓ_i / ∂φ
            grads = torch.autograd.grad(
                outputs=losses.sum(),
                inputs=logits,
                create_graph=False
            )[0]  # (N, m)
            targets = -grads.detach()
            modality_name = self.models[i][0]
             
            pred = self.models[i][1](data[modality_name], masks = mask[modality_name])
            loss = self.criterions[i](pred, targets)
            total_task_loss+=loss*len(batch)
            num_samples+=len(batch)
            if split=="train":
                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()
                
                
        metrics = {"task_loss_mse": total_task_loss.cpu().detach().numpy() / num_samples}
        return metrics   
             
    def enssemble_evaluate(self,fold_ind, dataloader, device, max_i=None):
        preds,times,events = [], [], []
        total_task_loss =0
        num_samples = 0
        
        for batch in dataloader:
            data, mask,  time, event = batch
            data = {modality :value.to(device) for modality, value in data.items()}
            logits = self._forward_all(data, mask,  max_i=max_i)
            eta_prev = self._compose_logits(logits, max_i=max_i)  # η = sum of prev models

            loss = self.criterions[0](eta_prev, time.to(device), event.to(dtype=torch.float32,device=device))
            preds.append(eta_prev)
            times.append(time)
            events.append(event)
            total_task_loss+=loss*len(batch)
            num_samples+=len(batch)

        metrics = {"task_loss_ensemble": total_task_loss.cpu().detach().numpy() / num_samples}
        preproc = self.preproc[next(iter(self.preproc))] if isinstance(self.preproc, dict) else self.preproc
        metrics.update(compute_survival_metrics( preds, torch.cat(times, dim=0), torch.cat(events, dim=0), cuts=preproc.get_hazard_cuts()))
            
        return  metrics
            
              
    def train(self, fold_ind : int):
        print("Train...")
        for i in range(len(self.models)):
            print(f"\nTraining model {i+1}/{len(self.models)}")
            if i == 0:
                for epoch in tqdm(range(self.cfg.base.n_epochs)):
                    
                    self.models[i][1].train()
                    train_metrics_fm = self.__loop__("train",fold_ind, self.dataloaders['train'], self.cfg.base.device)
                    train_metrics_fm.update({"epoch": epoch})
                    if self.cfg.base.log.logging:
                        wandb.log({f"train_first_model/fold_{fold_ind}/{key}" : value for key, value in train_metrics_fm.items()})
                    with torch.no_grad():    
                        val_metrics = self.__loop__("val",fold_ind, self.dataloaders['val'], self.cfg.base.device)
                        val_metrics.update({"epoch": epoch})    
                        if self.cfg.base.log.logging:
                            wandb.log({f"val_first_model/fold_{fold_ind}/{key}" : value for key, value in val_metrics.items()})        

            else:
                print("Run boosting training : ")
                for epoch in tqdm(range(self.cfg.base.n_epochs_boosting)):
                    self.models[i][1].train()
                    train_metrics_boost = self.__boosting_step__("train", fold_ind, self.dataloaders['train'], i, self.cfg.base.device)
                    train_metrics_boost.update({"epoch": epoch})
                    if self.cfg.base.log.logging:
                            wandb.log({f"train_bosting_model_{i}/fold_{fold_ind}/{key}" : value for key, value in train_metrics_boost.items()})
                    with torch.no_grad():
                        # val_metrics_boost = self.__boosting_step__("val", fold_ind, self.dataloaders['val'], i, self.cfg.base.device)
                        # val_metrics_boost.update({"epoch": epoch})    
                        # if self.cfg.base.log.logging:
                        #     wandb.log({f"val_boosting_model_{i}/fold_{fold_ind}/{key}" : value for key, value in val_metrics_boost.items()})
                        val_metrics_ens = self.enssemble_evaluate(fold_ind, self.dataloaders['val'], self.cfg.base.device, max_i=i)
                        val_metrics_ens.update({"epoch": epoch})    
                        if self.cfg.base.log.logging:
                            wandb.log({f"val_boosting_enssemble_model_{i}/fold_{fold_ind}/{key}" : value for key, value in val_metrics_ens.items()})
                    
        return val_metrics
        
    def evaluate(self, fold_ind : int):
        for i in range(len(self.modalities)):
             self.models[i][1].eval()
        with torch.no_grad():    
            test_metrics_intersection = None
            test_metrics = self.enssemble_evaluate(fold_ind, self.dataloaders['test'], self.cfg.base.device)
            if "test_intersection" in self.dataloaders:
                test_metrics_intersection = self.enssemble_evaluate(fold_ind, self.dataloaders['test_intersection'], self.cfg.base.device)
            if self.cfg.base.log.logging:
                wandb.log({f"test/fold_{fold_ind}/{key}" : value for key, value in test_metrics.items()})
                if "test_intersection" in self.dataloaders:
                    wandb.log({f"test_intersection/fold_{fold_ind}/{key}" : value for key, value in test_metrics_intersection.items()})

        return test_metrics, test_metrics_intersection 

        
        
        



 