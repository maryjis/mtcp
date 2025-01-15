
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
from ...preprocessor import BaseUnimodalPreprocessor
#  TODO one class -> like transfer scaler.fit_predict
from pathlib import Path
from src.unimodal.dna.datasets import DNAmDataset
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
                 n_intervals: int, var_threshold = 0.0, is_cluster_genes: bool = False , threshold: float =0):
        super().__init__(data_train, dataset_dir, n_intervals, StandardScaler,
                        {}, var_threshold, is_cluster_genes,threshold )
        
        self.train_dataset = DNAmDataset(data_train, dataset_dir)
        self.train_loaders = DataLoader(self.train_dataset)
        self.pipe =Pipeline(steps= [('var' , VarianceThreshold(var_threshold))])
       
    def get_scaling(self):
        raise NotImplementedError("DNAmPreprocessor doesn't support scaling")
    
