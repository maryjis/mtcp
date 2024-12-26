from torch.utils.data import Dataset, DataLoader
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
        