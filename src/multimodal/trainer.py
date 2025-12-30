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
from src.unimodal.rna.transforms import padded_transforms_with_scaling
from src.unimodal.dna.transforms import padded_transforms_scaling
from src.unimodal.cnv.transforms import padded_transforms_cnv_scaling
from src.unimodal.clinical.transforms import base_scaling
from src.unimodal.trainer import Trainer, UnimodalSurvivalTrainer, UnimodalMAETrainer
from src.multimodal.models import MultiMaeForPretraining, MultiMaeForSurvival
import torch
from src.utils import check_dir_exists, count_parameters, print_vit_sizes
from src.utils import ExperimentTracker
from pycox.models.loss import CoxPHLoss
from src.multimodal.losses import PairwiseRankingLoss
from src.unimodal.rna.transforms import PosFractionSampler
from ..evaluation import compute_survival_metrics
from itertools import chain

class MultiModalTrainer(Trainer):
    def __init__(self, splits: Dict[str,pd.DataFrame], cfg: DictConfig, tracker: ExperimentTracker, fold_index: int =0):
        self.cfg =cfg
        self.fold_index =fold_index
        print("self.cfg: ", self.cfg)
        self.preproc = self.initialise_preprocessing(splits, self.cfg.base.modalities)
        print(self.preproc)
        self.tracker = tracker
        self.best_model = None
        
 
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
    
    def __init__(self, splits: Dict[str,pd.DataFrame], cfg: DictConfig, tracker: ExperimentTracker, fold_index: int =0):
        super().__init__(splits, cfg, tracker, fold_index)
        # TODO MRI - done preprocess! 

        if cfg.model.get("rna_model", None):
            cfg.model.rna_model.size = cfg.model.rna_model.size if cfg.model.rna_model.get("size", None) else math.ceil(len(self.preproc["rna"].get_column_order()) /cfg.model.rna_model.patch_size)* cfg.model.rna_model.patch_size
        transforms = {"rna": padded_transforms_with_scaling(self.preproc["rna"].get_scaling(), cfg.model.rna_model.size, cfg.model.rna_model.is_padding) if cfg.model.get("rna_model", None) else None, 
                        "dnam" : padded_transforms_scaling(self.preproc["dnam"].get_scaling(), cfg.model.dnam_model.get("size", None),cfg.model.dnam_model.is_padding) if cfg.model.get("dnam_model", None) else None,
                        "cnv" : padded_transforms_cnv_scaling(self.preproc["cnv"].get_scaling(), cfg.model.cnv_model.get("size", None)) if "cnv" in self.preproc.keys() else None,
                        "mri" : None, "wsi" : None }

        self.datasets = self.initialise_datasets(splits, self.cfg.base.modalities, self.preproc, transforms)
        self.dataloaders = {split: DataLoader(self.datasets[split],shuffle=True if split == "train" else False, batch_size=cfg.base.batch_size, drop_last=True if split == "train" else False)
                            for split in splits.keys()}
        print("Device from config: ", cfg.base.device)
        self.model =self.initialise_models().to(cfg.base.device)
        print(self.model)
        print("Model device: ", next(self.model.parameters()).device)
        print("Model params count:", sum(p.numel() for p in self.model.parameters()))
        torch.cuda.reset_peak_memory_stats(device=cfg.base.device)
        print(f"gpu used {torch.cuda.max_memory_allocated(device=cfg.base.device)/1024/1024} Mb memory")
        self.initialise_loss()
    
    def initialise_loss(self):
        self.optimizer = AdamW(self.model.parameters(), **self.cfg.base.optimizer.params)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.base.n_epochs,**self.cfg.base.scheduler.params)
        
    def initialise_models(self):

        return MultiMaeForPretraining(ViTMAEConfig(**OmegaConf.to_container(self.cfg.model)), tracker = self.tracker,  preproc =self.preproc)

 
        
    def __loop__(self,split, fold_ind, dataloader, device):
        total_loss =0
        modality_losses = {}
        contrastive_losses ={}
        modality_count ={modality: 0 for modality in self.cfg.base.modalities}
        for batch in dataloader:
            data, masks = batch 
            data = {modality :value.to(device) for modality, value in data.items()} 
            outputs =self.model(data,masks)

            if split=="train":
                self.optimizer.zero_grad()
                outputs.loss[0].backward()
                self.optimizer.step()
            
            for modality, _ in data.items():
                if modality not in modality_losses:
                    modality_losses[modality] = 0
                modality_losses[modality] += outputs.loss[1][modality] * torch.sum(masks[modality])
                modality_count[modality] +=torch.sum(masks[modality])
            contrastive_loss = outputs.loss[2]

            total_loss+=outputs.loss[0]*masks[modality].shape[0]
            # print("masks sum: ",modality,  torch.sum(masks[modality]))
            # print("Batch lenth: ",data[modality].shape[0])

            if  contrastive_loss:
                for name, loss_value in contrastive_loss.items():
                    if name not in contrastive_losses:
                        contrastive_losses[name] = 0
                    contrastive_losses[name] += outputs.loss[2][name] 
                    
        metrics = {"mse_loss": total_loss.cpu().detach().numpy() / len(dataloader.dataset)}
        modalities = self.cfg.base.modalities
        for modality in modalities:
            metrics[f"mse_{modality}_loss"] = modality_losses[modality].cpu().detach().numpy() / modality_count[modality]
        for contrastive_name, contrastive_value in contrastive_losses.items():
            metrics[f"clip_{contrastive_name}_loss"] =contrastive_value.cpu().detach().numpy() / len(dataloader.dataset)
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
    def __init__(self, splits: Dict[str,pd.DataFrame], cfg: DictConfig, tracker: ExperimentTracker, fold_index: int =0):
        super().__init__(splits, cfg, tracker, fold_index)
        ## TODO MRI add transforms!!
        if cfg.model.get("rna_model", None):
            cfg.model.rna_model.size = cfg.model.rna_model.size if cfg.model.rna_model.get("size", None) else math.ceil(len(self.preproc["rna"].get_column_order()) /cfg.model.rna_model.patch_size)* cfg.model.rna_model.patch_size
        transforms = {"rna": padded_transforms_with_scaling(self.preproc["rna"].get_scaling(), cfg.model.rna_model.size,cfg.model.rna_model.is_padding) if cfg.model.get("rna_model", None) else None,
                      "dnam" : padded_transforms_scaling(self.preproc["dnam"].get_scaling(), cfg.model.dnam_model.get("size", None), cfg.model.dnam_model.is_padding) if cfg.model.get("dnam_model", None) else None,
                      "mri" : None, "wsi" : None,
                      "cnv" : padded_transforms_cnv_scaling(self.preproc["cnv"].get_scaling(), cfg.model.cnv_model.get("size", None)) if "cnv" in self.preproc.keys() else None,
                      "clinical" : base_scaling(self.preproc["clinical"].get_scaling() if "clinical" in self.preproc.keys() else None)}

        self.datasets = self.initialise_datasets(splits, self.cfg.base.modalities, self.preproc, transforms)
        self.dataloaders = {}
        for split in splits.keys():
            if split == "train":
                # берем события напрямую из датасета
                events = self.datasets["train"].data.event.values  # или .to_numpy(), если это pandas
                sampler = PosFractionSampler(events, batch_size=self.cfg.base.batch_size, pos_frac=1/10)
                if self.cfg.base.get("sampler", True):
                    print("sampler:")
                    self.dataloaders[split] = DataLoader(
                        self.datasets[split],
                        batch_sampler=sampler,
                        num_workers=self.cfg.base.get("num_workers", 0)
                    )
                else:
                    self.dataloaders[split] = DataLoader(
                        self.datasets[split],
                        shuffle=True,
                        batch_size=self.cfg.base.batch_size,
                        drop_last=True,
                        num_workers=self.cfg.base.get("num_workers", 0)
                    )
            else:
                self.dataloaders[split] = DataLoader(
                    self.datasets[split],
                    shuffle=False,
                    batch_size=1,
                    num_workers=self.cfg.base.get("num_workers", 0)
                )
        self.model = self.initialise_models().to(cfg.base.device)
        self.initialise_loss()
        print(self.model)
    
    def initialise_loss(self):
        self.criterion_logistic = NLLLogistiHazardLoss()    
        if self.cfg.base.loss.additional_loss =="PairwiseRankLoss":
            self.criterion_additional = PairwiseRankingLoss()
        elif self.cfg.base.loss.additional_loss =="CoxPHLoss":
            self.criterion_additional = CoxPHLoss()
        else:
            raise NotImplementedError(f"Such loss isn't implemented {self.cfg.base.loss.additional_loss}")
        
        if self.cfg.model.multimodal_fusion =="multiple_heads":
            self.optimizer = AdamW(
                        params=[
                            *self.model.model.encoder_fusion_strategy.parameters(),
                            *self.model.fusion_strategy.parameters(),
                            *self.model.projections["multimodal"].parameters(),
                        ],
                        **self.cfg.base.optimizer.params
                    )
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.base.n_epochs,**self.cfg.base.scheduler.params)

            self.criterion_logistic_unimodal = NLLLogistiHazardLoss()    
            self.criterion_additional_unimodal = CoxPHLoss()

            param_groups = []

            for m in self.cfg.base.modalities:
                lr_m = self.cfg.base.optimizer_unimodal[m].params.lr

                param_groups.append({
                    "params": chain(
                        self.model.model.encoders[m].parameters(),
                        self.model.projections[m].parameters(),
                    ),
                    "lr": lr_m,
                    "weight_decay": self.cfg.base.optimizer_unimodal[m].params.weight_decay,
                })

            self.optimizer_unimodal = AdamW(
                    param_groups
                )

            # self.optimizer_unimodal = AdamW(
            #     params=chain(
            #         # unimodal encoders
            #         *(self.model.model.encoders[m].parameters() for m in self.cfg.base.modalities),
            #         # unimodal heads
            #         *(self.model.projections[m].parameters() for m in self.cfg.base.modalities),
            #     ),
            #     **self.cfg.base.optimizer_unimodal.params
            # )

            self.scheduler_unimodal = CosineAnnealingLR(self.optimizer_unimodal, T_max=self.cfg.base.n_epochs,**self.cfg.base.scheduler_unimodal.params)
        else:
            self.optimizer = AdamW(self.model.parameters(), **self.cfg.base.optimizer.params)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.base.n_epochs,**self.cfg.base.scheduler.params)

    def initialise_models(self):

        return MultiMaeForSurvival(ViTMAEConfig(**OmegaConf.to_container(self.cfg.model)), tracker = self.tracker, preproc =self.preproc)

    def __loop__(self,split, fold_ind, dataloader, device):
        total_task_loss =0
        num_samples = 0
        times,events = [], []
        preds = defaultdict(list)

        for batch in dataloader:
            data, mask,  time, event = batch 
            data = {modality :value.to(device) for modality, value in data.items()} if isinstance(data, dict) else data.to(device)
            batch_size = (next(iter(data.values())).shape[0] if isinstance(data, dict) else data.shape[0])
            if split=="train":
                self.optimizer.zero_grad()

            pred_values = 0
            if self.cfg.model.multimodal_fusion =="multiple_heads":
                unimodal_loss = 0
                if split=="train":
                    self.optimizer_unimodal.zero_grad()

                outputs_dict =self.model.forward_unimodal_encoders(data, masks = mask) # here is a dictinary -> modality -outputs
                
                # Train unimodal encoders
                for output_modality, outputs in outputs_dict.items():
                    print("Through: ", output_modality)
                    loss1 = self.criterion_logistic_unimodal(outputs, time.to(device), event.to(dtype=torch.float32,device=device))
                
                    haz = torch.sigmoid(outputs)
                    cum_hazard = -torch.log1p(-haz + 1e-7).cumsum(1)
                    risk = cum_hazard[:, -1]
            
                    loss2 = self.criterion_additional_unimodal(risk, time.to(device), event.to(dtype=torch.float32,device=device))
            
                    loss  =  self.cfg.base.loss.alpha *  loss1  + (1- self.cfg.base.loss.alpha) * loss2
                    unimodal_loss+=loss
                    # # Backpropagation
                    # if split=="train":
                    #     loss.backward()
                    preds[output_modality].append(outputs.detach().cpu())
                    pred_values+=outputs
                    total_task_loss+=loss*batch_size
                    num_samples+=batch_size
                if split=="train":
                    unimodal_loss.backward()
                    self.optimizer_unimodal.step()

            # Train multimodal encoder
            outputs_multimodal =self.model(data, masks = mask)
            haz = torch.sigmoid(outputs_multimodal)
            cum_hazard = -torch.log1p(-haz + 1e-7).cumsum(1)
            risk = cum_hazard[:, -1]

            loss1 = self.criterion_logistic(outputs_multimodal, time.to(device), event.to(dtype=torch.float32,device=device))
            loss2 = self.criterion_additional(risk, time.to(device), event.to(dtype=torch.float32,device=device))
            loss  =  self.cfg.base.loss.alpha *  loss1  + (1- self.cfg.base.loss.alpha) * loss2
    
            times.append(time.cpu())
            events.append(event.cpu())
            preds["multimodal"].append(outputs_multimodal.detach().cpu())
            pred_values += outputs_multimodal
            #preds["mean_val"].append(pred_values.detach().cpu() / len(outputs_dict.keys())+1)
             # Backpropagation
            if split=="train":
                loss.backward()
                self.optimizer.step()
        
        metrics = {"task_loss": total_task_loss.cpu().detach().numpy() / num_samples}
        if split!="train":
            preproc = self.preproc[next(iter(self.preproc))] if isinstance(self.preproc, dict) else self.preproc
            metrics.update(compute_survival_metrics( preds, torch.cat(times, dim=0), torch.cat(events, dim=0), cuts=preproc.get_hazard_cuts()))
            if len(preds.keys())>1:
                for key in preds.keys():
                    if key!= "mean_val":
                        metrics_modality=compute_survival_metrics( preds[key], torch.cat(times, dim=0), torch.cat(events, dim=0), cuts=preproc.get_hazard_cuts())
                        metrics.update({f"{key}_{name}" : value for name, value in metrics_modality.items()})
        else:
            self.scheduler.step()
            if self.cfg.model.multimodal_fusion =="multiple_heads":
                self.scheduler_unimodal.step()
            
        return  metrics 

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
                                                 transform = transforms, is_hazard_logits = True,
                                                 debug_mode =self.cfg.data.rna.debug_mode)
        return concat_dataset
    


 
