from torch.utils.data import Dataset, DataLoader
import pandas as pd


class BaseDataset(Dataset):
    
    def __init__(self, data, root_dir = None, transform = None , is_hazard_logits = False, return_mask = False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory to RNA csv files
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        self.is_hazard_logits = is_hazard_logits
        self.return_mask = return_mask
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
         raise NotImplementedError("This is absctract class")