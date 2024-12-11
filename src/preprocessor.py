from pycox.models import LogisticHazard
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
#  TODO one class -> like transfer scaler.fit_predict



class BaseUnimodalPreprocessor(object):
    
    def __init__(self, data_train :pd.DataFrame, n_intervals: int):
        
        self.data_train = data_train
        self.hazard = LogisticHazard.label_transform(n_intervals)
        
        
    def fit(self):
        self.hazard.fit(self.data_train['time'], self.data_train['event']) 
    
    def transform_labels(self, dataset):
        dataset['new_time'], dataset['new_event'] =self.hazard.transform(dataset['time'], dataset['event'])
        return dataset
    
    def get_hazard_cuts(self):
        return self.hazard.cuts
        