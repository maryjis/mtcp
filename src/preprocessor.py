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
        
        
class QuantilePreprocessor(object):
    def __init__(self, data_train: pd.DataFrame, n_intervals: int):
        self.data_train = data_train
        self.n_intervals = n_intervals
        self.quantile_bins = None

    def fit(self):
        self.data_train = self.data_train.copy()
        self.data_train['time_bin'], self.quantile_bins = pd.qcut(
            self.data_train['time'],
            q=self.n_intervals,
            labels=False,
            retbins=True,
            duplicates='drop'
        )

    def transform_labels(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset.copy()
        min_bin, max_bin = self.quantile_bins[0], self.quantile_bins[-1]
        eps = 1e-6  # малое смещение, чтобы избежать попадания на границу

        # Обрезаем значения внутри допустимого диапазона
        dataset['clipped_time'] = dataset['time'].clip(lower=min_bin + eps, upper=max_bin - eps)

        # Присваиваем интервалы
        dataset['new_time'] = pd.cut(
            dataset['clipped_time'],
            bins=self.quantile_bins,
            labels=False,
            include_lowest=True,
            right=True
        )

        # Жёстко присваиваем интервалы для значений вне диапазона
        dataset.loc[dataset['time'] <= min_bin, 'new_time'] = 0
        dataset.loc[dataset['time'] >= max_bin, 'new_time'] = self.n_intervals - 1

        dataset['new_time'] = dataset['new_time'].astype(int)
        dataset['new_event'] = dataset['event']
        return dataset

    def get_hazard_cuts(self):
        return self.quantile_bins