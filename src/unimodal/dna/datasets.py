
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from ...datasets import BaseDataset
import torch
import numpy as np

class DNAmDataset(BaseDataset):
    """RNA dataset."""

    def __init__(self, data_split, dataset_file, transform = None, 
                 is_hazard_logits = False,
                 column_order = None,
                 return_mask = True):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory to RNA csv files
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(data_split, dataset_file, transform, is_hazard_logits, return_mask)
        self.dna_dataset = pd.read_csv(dataset_file)
        self.column_order = column_order
        
        if isinstance(column_order, pd.Index): 
            self.column_order = self.column_order.append(pd.Index(["file_id"]))
            self.dna_dataset = self.dna_dataset[self.column_order]
        elif isinstance(column_order,np.ndarray):
            self.column_order = np.append(self.column_order, "file_id")
            self.dna_dataset = self.dna_dataset[self.column_order]
        else:   
            self.column_order = self.dna_dataset.columns[:-1]

    def len(self):
        return self.data.shape[0]
    
    def get_column_names(self):
        return self.column_order
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        mask = False
        
        if not pd.isna(sample["DNAm"]):
            
            sample =self.dna_dataset.loc[self.dna_dataset["file_id"]==sample["DNAm"]]
            if sample.empty:
                return torch.zeros((1, self.dna_dataset.shape[1]-1)).float(), mask
            else:
                sample =sample.iloc[0, :-1].astype(np.float32).fillna(0)
   
                sample = sample.values.reshape(1, -1)
         
            mask = True
            if self.transform:
                sample = self.transform(sample)
                
            sample = torch.from_numpy(sample)
            return sample.float(), mask
        else:
            sample = torch.zeros((1, self.dna_dataset.shape[1]-1))
            if self.transform:
                sample = self.transform(sample)
                sample = torch.from_numpy(sample)
           
            return sample.float(), mask
            
        
class DNAmSurvivalDataset(DNAmDataset):
        def __init__(self, data_split, dataset_dir, transform = None, 
            is_hazard_logits = False, column_order = None, return_mask =True):
            super().__init__(data_split, dataset_dir, transform = transform, is_hazard_logits = is_hazard_logits, column_order = column_order, return_mask=return_mask)
            


        def __getitem__(self, idx):
            sample, mask = super().__getitem__(idx)
            if self.is_hazard_logits:
                if self.return_mask ==True:
                    return sample,mask,  self.data.iloc[idx]['new_time'], self.data.iloc[idx]['new_event']
                else:
                    return sample, self.data.iloc[idx]['new_time'], self.data.iloc[idx]['new_event']
            else:
                if self.return_mask ==True: 
                    return sample,mask, self.data.iloc[idx]['time'], self.data.iloc[idx]['event']
                else:
                    return sample, self.data.iloc[idx]['time'], self.data.iloc[idx]['event']
                
