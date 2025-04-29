from src.unimodal.rna.transforms import Padding
from torchvision.transforms import ToTensor
from torch.nn import Identity
import numpy as np
from torchvision import transforms
from sklearn.base import BaseEstimator, TransformerMixin
from src.unimodal.rna.transforms import UpperQuartileNormalizer
from torchvision.transforms import ToTensor
import pandas as pd

class Scale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, standart_scaler):
        self.standart_scaler =standart_scaler
        

    def __call__(self, sample):
        sample =self.standart_scaler.transform(sample)

        return sample
    
import torch
from torchvision import transforms
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class Log2RatioTransformSk(BaseEstimator, TransformerMixin):
    def __init__(self, base_cn=2, min_log2=-5):
        self.base_cn = base_cn
        self.min_log2 = min_log2
    
    def fit(self, X, y=None):
        """
        Метод fit не требует обучения для этой трансформации,
        но он обязателен для использования класса в scikit-learn pipeline.
        """
        return self
    
    def transform(self, X):
        """
        Преобразует данные, вычисляя log2(CN / base_cn) с обработкой нулевых и NaN значений:
        - CN == 0 → log2 = min_log2
        - CN == NaN → log2 = NaN

        Поддерживает numpy.ndarray и pandas.DataFrame.
        """
        is_dataframe = isinstance(X, pd.DataFrame)

        if is_dataframe:
            values = X.values
        elif isinstance(X, np.ndarray):
            values = X
        else:
            raise TypeError(f"Unsupported data type: {type(X)}. Expected numpy.ndarray or pandas.DataFrame.")
        
        zero_mask = values == 0
        log2ratio = np.log2(values / self.base_cn)
        log2ratio[zero_mask] = self.min_log2
        
        if is_dataframe:
            return pd.DataFrame(log2ratio, columns=X.columns, index=X.index)
        else:
            return log2ratio
    
    
class Log2RatioTransform:
    def __init__(self, base_cn=2, min_log2=-5):
        self.base_cn = base_cn
        self.min_log2 = min_log2
    
    def __call__(self, cn_tensor):
        """
        Вычисление log2(CN / base_cn) с обработкой нулевых и NaN значений:
        - CN = 0 → log2 = min_log2
        - CN = NaN → log2 = NaN
        """
        # Места, где CN == 0
        zero_mask = cn_tensor == 0
        
        # Места, где CN NaN (для NaN мы используем torch.isnan)
        #nan_mask = torch.isnan(cn_tensor)
        
        # Вычисление log2ratio: log2(CN / base_cn) для всех ненулевых и не NaN значений
        log2ratio = torch.log2(cn_tensor / self.base_cn)
        
        # Заменяем NaN на NaN, а 0 на min_log2
        log2ratio[zero_mask] = self.min_log2
        # log2ratio[nan_mask] = float('nan')  # или torch.nan, если PyTorch поддерживает
        
        return log2ratio
    
def padded_transforms_cnv_scaling(standart_scaler, size=None):

    if standart_scaler is not None:
        return transforms.Compose([
                                Log2RatioTransform(),
                                Scale(UpperQuartileNormalizer(quantile=50)),
                                Scale(standart_scaler),
                                Padding(size)])
    else:
        return transforms.Compose([

                                Log2RatioTransform(),
                                Scale(UpperQuartileNormalizer(quantile=50)),
                                Padding(size)])