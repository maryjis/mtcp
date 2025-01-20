import pandas as pd
from ..datasets import BaseDataset

class MultimodalDataset(BaseDataset):
    
    def __init__(self, data_split, dataset_dir, datasets, transform = None, 
                 is_hazard_logits = False, return_mask = True):
        # Todo убрать transform, добавить return_mask в  RNA
        super().__init__(data_split, dataset_dir, transform, is_hazard_logits, return_mask)
        self.datasets = datasets
        
    def __getitem__(self, idx):
        sample, masks = {} , {}
        for modality_name, dataset in self.datasets:
            modality_sample =dataset.__getitem__(idx)
            sample[modality_name] = modality_sample[0]
            masks[modality_name] = modality_sample[1]
        return sample, masks
    
    
class MultimodalSurvivalDataset(MultimodalDataset):
    
    def __init__(self, data_split, dataset_dir, datasets, transform = None, 
                 is_hazard_logits = True, return_mask = True):
        # Todo убрать transform, добавить return_mask в  RNA
        super().__init__(data_split, dataset_dir, datasets, transform, 
                 is_hazard_logits, return_mask)
        
        
    def __getitem__(self, idx):
        sample = self.datasets[0][1].__getitem__(idx)
        
        times, events = sample[2], sample[3]
        sample, masks = super().__getitem__(idx)
    
        return sample, masks, times, events 
        