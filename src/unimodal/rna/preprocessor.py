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
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from src.unimodal.rna.transforms import log_transform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin
from sklearn.feature_selection import VarianceThreshold

class RNAPreprocessor(BaseUnimodalPreprocessor):
    
    def __init__(self, data_train :pd.DataFrame, dataset_dir : Path,
                 n_intervals: int, scaling_method: TransformerMixin, scaling_prams: dict = {},
                 var_threshold = 0.0, is_cluster_genes: bool = False , threshold: float =0):
        super().__init__(data_train, n_intervals)
        self.data_train = data_train
        self.train_dataset = RNADataset(data_train, dataset_dir)
        self.train_loaders = DataLoader(self.train_dataset)
        
        self.pipe = Pipeline(steps=[('log', FunctionTransformer(log_transform)) , ('scaler', scaling_method(**scaling_prams)), ('var' , VarianceThreshold(var_threshold))])
        self.is_cluster_genes =is_cluster_genes
        self.column_order = None
        self.threshold =threshold
    
        
    def fit(self):
        print("RNA Preprocessing......")
        super().fit() 
        data = self.load_data(self.train_loaders)
        data = pd.DataFrame(data = data, columns =self.train_dataset.get_column_names())
        
        print(data)
        self.pipe.fit(data)
        print( "Number of features: ", self.pipe['var'].get_feature_names_out().shape)
        
        
        if self.is_cluster_genes:
            print("Clustering genes:")
            data = self.pipe.transform(data)
            dataset = pd.DataFrame(data = data, columns = self.pipe['var'].get_feature_names_out())
            
            correlations = dataset.corr()
            dissimilarity = 1 - abs(correlations)
            Z = linkage(squareform(dissimilarity), 'complete')
            # Clusterize the data
            labels = fcluster(Z, self.threshold, criterion='distance')

            
            labels_order = np.argsort(labels)
        
            self.column_order = self.train_dataset.get_column_names()[labels_order]
            print("--------------------------------------------------------")
            print("New columns: ", self.column_order)
        
    
    def transform_labels(self, dataset):
        dataset =super().transform_labels(dataset)
        return dataset 
    
    # def fit_transform(self, dataset):
    #     self.fit()
    #     return self.transform_labels(dataset)
        
        
    def get_scaling(self):
        return self.pipe['scaler']
    
    def get_column_order(self):
        if self.column_order is None:
            return self.pipe['var'].get_feature_names_out()
        else:
            return self.column_order
        
    @staticmethod
    def load_data(dataloader):  
        data =[] 
        for batch in dataloader:
            data.append(batch[0].cpu().numpy()[:,0,:])
        data = np.concatenate(data)
        return data