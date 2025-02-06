
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from ...datasets import BaseDataset
import torch
import numpy as np

class ClinicalDataset(BaseDataset):
    """RNA dataset."""

    def __init__(self, data_split,root_dir, selected_columns, transform = None, 
                 is_hazard_logits = False,
                 return_mask = True):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory to RNA csv files
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(data_split, root_dir, transform, is_hazard_logits, return_mask)
        self.selected_columns = selected_columns
        


    def len(self):
        return self.data.shape[0]
    
    def get_column_names(self):
        return self.column_order
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        mask = False
        sample = sample[self.selected_columns]
        return sample.float(), mask
 
            
        
class ClinicalSurvivalDataset(ClinicalDataset):
        def __init__(self, data_split, dataset_dir,selected_columns, transform = None, 
            is_hazard_logits = False, return_mask =True):
            super().__init__(data_split, dataset_dir, selected_columns= selected_columns,transform = transform, is_hazard_logits = is_hazard_logits, return_mask=return_mask)
            
            # TODO подумать как тут лучше сделать выгрузку для мультимодальных данных 
            # TODO добавить return_mask
            # if not return_nan:
            #     self.data  = self.data.loc[self.data['RNA'].isin(self.rna_dataset['file_id'].to_list())]

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
                