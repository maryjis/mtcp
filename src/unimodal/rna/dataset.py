
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from ...datasets import BaseDataset
import torch
import numpy as np

class RNADataset(BaseDataset):
    """RNA dataset."""

    def __init__(self, data_split, dataset_dir, transform = None, is_hazard_logits = False, column_order =None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory to RNA csv files
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(data_split, dataset_dir, transform, is_hazard_logits)
        self.rna_dataset = pd.read_csv(dataset_dir)
        self.rna_dataset = self.rna_dataset.loc[self.rna_dataset['file_id'].isin(data_split['RNA'].to_list())].reset_index(drop=True)

        if isinstance(column_order, pd.Index) or isinstance(column_order,np.ndarray):
            print(column_order)  
            self.column_order = column_order
            self.rna_dataset = self.rna_dataset[self.column_order]
        else:
            self.column_order = self.rna_dataset.columns[:-1]
            self.rna_dataset =self.rna_dataset.iloc[:, :-1]

    def len(self):
        return self.rna_dataset.shape[0]
    
    def get_column_names(self):
        return self.column_order
    
    def __getitem__(self, idx):
       
        sample =self.rna_dataset.iloc[idx, :].values.reshape(1, -1).astype(np.float32)
        
        if self.transform:
            sample = self.transform(sample)
              
        sample = torch.from_numpy(sample)
        if self.is_hazard_logits:
            return sample.float(), self.data.iloc[idx]['new_time'], self.data.iloc[idx]['new_event']
        else: 
            return sample.float(), self.data.iloc[idx]['time'], self.data.iloc[idx]['event']
