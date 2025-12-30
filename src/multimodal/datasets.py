import pandas as pd
from ..datasets import BaseDataset

class MultimodalDataset(BaseDataset):
    
    def __init__(self, data_split, dataset_dir, datasets, transform = None, 
                 is_hazard_logits = False, return_mask = True, debug_mode=False):
        
        super().__init__(data_split, dataset_dir, transform, is_hazard_logits, return_mask)
        self.datasets = datasets
        self.debug_mode = debug_mode
        
    def __getitem__(self, idx):
        if not self.debug_mode: 
            sample, masks = {} , {}
            for modality_name, dataset in self.datasets:
                modality_sample =dataset.__getitem__(idx)
                sample[modality_name] = modality_sample[0]
                masks[modality_name] = modality_sample[1]
            return sample, masks
        else:
            sample, masks, file_ids = {} , {}, {}
            for modality_name, dataset in self.datasets:
                modality_sample =dataset.__getitem__(idx)
                sample[modality_name] = modality_sample[1]
                masks[modality_name] = modality_sample[2]
                file_ids[modality_name] = modality_sample[0]
            return file_ids,sample, masks
    
    
class MultimodalSurvivalDataset(MultimodalDataset):
    
    def __init__(self, data_split, dataset_dir, datasets, transform = None, 
                 is_hazard_logits = True, return_mask = True, debug_mode =False):
        
        super().__init__(data_split, dataset_dir, datasets, transform, 
                 is_hazard_logits, return_mask,debug_mode)
        
        
    def __getitem__(self, idx):
        sample = self.datasets[0][1].__getitem__(idx)
        
        
        if not self.debug_mode: 
            times, events = sample[2], sample[3]
            sample, masks = super().__getitem__(idx)
            return sample, masks, times, events 
        else:
            times, events = sample[3], sample[4]
            file_ids, sample, masks = super().__getitem__(idx)
            return file_ids, sample, masks, times, events 
        
        
