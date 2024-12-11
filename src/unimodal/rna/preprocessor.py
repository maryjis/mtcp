from pycox.models import LogisticHazard
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
from ...preprocessor import BaseUnimodalPreprocessor
#  TODO one class -> like transfer scaler.fit_predict
from pathlib import Path
from src.unimodal.rna.dataset import RNADataset
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RNAPreprocessor(BaseUnimodalPreprocessor):
    
    def __init__(self, data_train :pd.DataFrame, root_dir : Path, n_intervals: int):
        super().__init__(data_train, n_intervals)
        self.data_train = data_train
        self.train_dataset = RNADataset(data_train, root_dir)
        self.train_loaders = DataLoader(self.train_dataset)
        self.standart_scaler = StandardScaler()
    
        
    def fit(self):
        super().fit() 
        data = self.load_data(self.train_loaders)
        self.standart_scaler.fit(data)
    
    def transform_labels(self, dataset):
        dataset =super().transform_labels(dataset)
        return dataset 
    
    def fit_transform(self, dataset):
        self.fit()
        return self.transform(dataset)
        
        
    def get_scaling(self):
        return self.standart_scaler
        
    @staticmethod
    def load_data(dataloader):  
        data =[] 
        for batch in dataloader:
            data.append(batch[0].cpu().numpy()[:,0,:])
        data = np.concatenate(data)
        return data