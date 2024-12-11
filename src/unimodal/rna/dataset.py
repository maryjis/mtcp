
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from ...datasets import BaseDataset
import torch

class RNADataset(BaseDataset):
    """RNA dataset."""

    def __init__(self, data, root_dir, transform = None, is_hazard_logits = False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory to RNA csv files
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(data, root_dir, transform, is_hazard_logits)

    def __getitem__(self, idx):
       
        sample_path =self.data.iloc[idx]['RNA']
        
        sample =pd.read_csv(self.root_dir+sample_path)["fpkm_uq_unstranded"].values.reshape(1, -1)
        
        if self.transform:
            sample = self.transform(sample)
            
        sample = torch.from_numpy(sample)
        
        if self.is_hazard_logits:
            return sample.float(), self.data.iloc[idx]['new_time'], self.data.iloc[idx]['new_event']
        else: 
            return sample.float(), self.data.iloc[idx]['time'], self.data.iloc[idx]['event']
