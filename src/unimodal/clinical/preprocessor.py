from pycox.models import LogisticHazard
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
from ...preprocessor import BaseUnimodalPreprocessor
#  TODO one class -> like transfer scaler.fit_predict
from pathlib import Path
from src.unimodal.clinical.datasets import ClinicalDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

class ClinicalPreprocessor(BaseUnimodalPreprocessor):
    
    def __init__(self, data_train :pd.DataFrame, dataset_dir : Path, selected_columns : dict,
                 n_intervals: int, scaling_method: TransformerMixin, scaling_prams: dict = {}):
        super().__init__(data_train, n_intervals)
        self.data_train = data_train
        self.selected_columns =selected_columns
        self.train_dataset = ClinicalDataset(data_train, dataset_dir, selected_columns)
        self.train_loaders = DataLoader(self.train_dataset)
        
        self.pipe = Pipeline([('scaler', scaling_method(**scaling_prams))])
     
        
    def fit(self):
        print("Clinical Preprocessing......")
        super().fit() 
        data = self.load_data(self.train_loaders)
        self.pipe.fit(data)
          
    
    def transform_labels(self, dataset):
        dataset =super().transform_labels(dataset)
        return dataset 
    
        
    def get_scaling(self):
        return self.pipe['scaler']
    
        
    @staticmethod
    def load_data(dataloader):  
        data =[] 
        for batch in dataloader:
            data.append(batch[0].cpu().numpy()[:,0,:])
        data = np.concatenate(data)
        return data