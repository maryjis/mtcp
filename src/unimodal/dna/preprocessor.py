
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
from ...preprocessor import BaseUnimodalPreprocessor
#  TODO one class -> like transfer scaler.fit_predict
from pathlib import Path
from src.unimodal.rna.dataset import OmicsDataset
from src.unimodal.rna.preprocessor import RNAPreprocessor
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from src.unimodal.rna.transforms import log_transform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.base import TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler,Normalizer


class DNAmPreprocessor(RNAPreprocessor):
    
    def __init__(self, data_train :pd.DataFrame, dataset_dir : Path,
                 n_intervals: int, var_threshold = 0.0, is_cluster_genes: bool = False , threshold: float =0,
                 is_hierarchical_cluster: bool =False, 
                 scaling_method= None, scaling_params = {}, project_ids = []):
        super().__init__(data_train, dataset_dir, n_intervals, scaling_method,
                        scaling_params, var_threshold, is_cluster_genes,threshold,is_hierarchical_cluster,project_ids)
        
        self.train_dataset = OmicsDataset(data_train, dataset_dir, column_name='DNAm',  project_ids = project_ids)
        self.train_loaders = DataLoader(self.train_dataset)
        self.pipe =Pipeline(steps= [('var' , VarianceThreshold(var_threshold))])
       

    
