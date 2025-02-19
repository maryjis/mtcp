
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from ...datasets import BaseDataset
import torch
import numpy as np

class RNADataset(BaseDataset):
    """RNA dataset."""

    def __init__(self, data_split, dataset_file, transform = None, 
                 is_hazard_logits = False,
                 column_order = None,
                 return_mask = True,
                 debug_mode = False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory to RNA csv files
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(data_split, dataset_file, transform, is_hazard_logits, return_mask)
        self.rna_dataset = pd.read_csv(dataset_file)
        print(self.rna_dataset["file_id"])
        self.column_order = column_order
        print("Column order type: ", type(self.column_order))
        self.debug_mode = debug_mode
        if isinstance(column_order, pd.Index): 
            
            self.column_order =self.column_order.append(pd.Index(["file_id"]))
            self.rna_dataset = self.rna_dataset[self.column_order]
        elif isinstance(column_order,np.ndarray):
            self.column_order = np.append(self.column_order, "file_id")
            self.rna_dataset = self.rna_dataset[self.column_order]
        else:   
            self.column_order = self.rna_dataset.columns[:-1]

    def len(self):
        return self.data.shape[0]
    
    def get_column_names(self):
        return self.column_order
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        mask = False
        file_id = None
        if not pd.isna(sample["RNA"]):
            name_rna =sample["RNA"]
            sample =self.rna_dataset.loc[self.rna_dataset["file_id"]==sample["RNA"]]
    
            if sample.empty:
                return torch.zeros((1, self.rna_dataset.shape[1]-1)).float(), mask
            else:
                file_id = sample["file_id"].values[0]
                sample = sample.values[0, :-1].reshape(1, -1).astype(np.float32)
         
            mask = True
            if self.transform:
                sample = self.transform(sample)
                
            sample = torch.from_numpy(sample)
            if self.debug_mode:
                return file_id,  sample.float(), mask
            
            return sample.float(), mask
        else:
            sample = torch.zeros((1, self.rna_dataset.shape[1]-1)).float()
            if self.transform:
                sample = self.transform(sample)
                sample = torch.from_numpy(sample)
                
            if self.debug_mode:
                return file_id,  sample.float(), mask
            return sample.float(), mask
        
class RNASurvivalDataset(RNADataset):
        def __init__(self, data_split, dataset_dir, transform = None, 
            is_hazard_logits = False, column_order = None, return_mask =True, debug_mode =False):
            super().__init__(data_split, dataset_dir,
                             transform = transform, 
                             is_hazard_logits = is_hazard_logits, 
                             column_order = column_order, 
                             return_mask=return_mask,
                             debug_mode = debug_mode)
            
            # TODO подумать как тут лучше сделать выгрузку для мультимодальных данных 
            # TODO добавить return_mask
            # if not return_nan:
            #     self.data  = self.data.loc[self.data['RNA'].isin(self.rna_dataset['file_id'].to_list())]

        def __getitem__(self, idx):
            if self.debug_mode:
                file_id, sample, mask = super().__getitem__(idx)
            else:
                sample, mask = super().__getitem__(idx)
            if self.is_hazard_logits:
                if self.return_mask ==True:
                    if self.debug_mode:
                        return file_id, sample.float(),mask,  self.data.iloc[idx]['new_time'], self.data.iloc[idx]['new_event']
                    return sample.float(),mask,  self.data.iloc[idx]['new_time'], self.data.iloc[idx]['new_event']
                else:
                    if self.debug_mode:
                        return file_id, sample.float(),  self.data.iloc[idx]['new_time'], self.data.iloc[idx]['new_event']
                    return sample.float(), self.data.iloc[idx]['new_time'], self.data.iloc[idx]['new_event']
            else:
                if self.return_mask ==True: 
                    return sample.float(),mask, self.data.iloc[idx]['time'], self.data.iloc[idx]['event']
                else:
                    return sample.float(), self.data.iloc[idx]['time'], self.data.iloc[idx]['event']
                