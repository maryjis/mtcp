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
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster,leaves_list
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from src.unimodal.rna.transforms import UpperQuartileNormalizer, TorchQuantileTransformer
from sklearn.preprocessing import QuantileTransformer
import os

class RNAPreprocessor(BaseUnimodalPreprocessor):
    
    def __init__(self, data_train :pd.DataFrame, dataset_dir : Path,
                 n_intervals: int, scaling_method: TransformerMixin, scaling_prams: dict = {},
                 var_threshold = 0.0, is_cluster_genes: bool = False , threshold: float =0, is_hierarchical_cluster: bool =False):
        super().__init__(data_train, n_intervals)
        self.data_train = data_train
        self.train_dataset = RNADataset(data_train, dataset_dir)
        self.train_loaders = DataLoader(self.train_dataset)
        
        self.pipe = Pipeline(steps=[('log', FunctionTransformer(log_transform)) ,
                                    ('scaler1',UpperQuartileNormalizer(quantile=50)), 
                                    ('var' , VarianceThreshold(var_threshold))])
        
        self.scaling_method = None
        if scaling_method is not None:
            # self.scaling_method = scaling_method(**scaling_prams)
            self.scaling_method =Pipeline(steps=[('scaler2', TorchQuantileTransformer(n_quantiles=1000, output_distribution='normal')), ('scaler3', scaling_method(**scaling_prams))])
        # if isinstance(scaling_method, StandardScaler):
        print("!!!!!scaler: ", self.scaling_method)
        #self.scaling_method.set_output(transform='pandas')
        self.is_cluster_genes =is_cluster_genes
        self.column_order = None
        self.threshold =threshold
        self.is_hierarchical_cluster =is_hierarchical_cluster
    
    def save_list_to_file(self, filename="../data/selected_genes/columns_set.txt"):
        """
        Save a list of strings to a file.
        If file already exists, increment filename.
        
        Parameters:
        - data: list of strings
        - filename: base filename (default "file.txt")
        """
        
        genes = self.pipe['var'].get_feature_names_out()
        base, ext = os.path.splitext(filename)
        counter = 1
        new_filename = filename

        while os.path.exists(new_filename):
            new_filename = f"{base}_{counter}{ext}"
            counter += 1

        with open(new_filename, "w", encoding="utf-8") as f:
            for line in genes:
                f.write(line + "\n")
        
        print(f"Saved to: {new_filename}")
        return new_filename
    
    def fit(self):
        print("RNA Preprocessing......")
        super().fit() 
        data = self.load_data(self.train_loaders)
        data = pd.DataFrame(data = data, columns =self.train_dataset.get_column_names())
        
        print(data)
        self.pipe.fit(data)
        print( "Number of features: ", self.pipe['var'].get_feature_names_out().shape)
        # self.save_list_to_file()
        data = self.pipe.transform(data)
        if self.scaling_method is not None:
            data = self.scaling_method.fit_transform(data)

        dataset = pd.DataFrame(data = data, columns = self.pipe['var'].get_feature_names_out())
        
        print("Dataset after scaling", dataset)
        if self.is_cluster_genes:
            correlations = dataset.corr()
            dissimilarity = 1 - abs(correlations)
            Z = linkage(squareform(dissimilarity), 'complete')
            # Clusterize the data
            
            if self.is_hierarchical_cluster:
                print("self.is_hierarchical_cluster: ", self.is_hierarchical_cluster)
                labels_order = leaves_list(Z)
            else:
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
        return self.scaling_method
    
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