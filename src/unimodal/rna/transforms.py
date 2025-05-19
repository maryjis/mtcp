import numpy as np
from torchvision import transforms
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn.functional as F
from scipy.stats import norm
import torch
import numpy as np
import pandas as pd

class TorchQuantileTransformer:
    """
    Преобразователь квантилей на PyTorch, аналогичный sklearn.preprocessing.QuantileTransformer.
    
    Для каждого признака:
      1. Вычисляются эмпирические квантильные границы по обучающей выборке.
      2. Для каждого значения определяется его квантиль (в диапазоне [0,1]) с помощью torch.searchsorted.
      3. Если output_distribution='normal', значения отображаются в стандартное нормальное распределение через:
            norm.ppf(y) ≈ sqrt(2) * erfinv(2*y - 1)
    
    Параметры
    ---------
    n_quantiles : int, default=1000
        Количество квантилей, используемых для аппроксимации эмпирической CDF.
        Если n_quantiles больше числа образцов, используется число образцов.
    output_distribution : {'uniform', 'normal'}, default='uniform'
        Выходное распределение после преобразования.
    """
    def __init__(self, n_quantiles=10000, output_distribution='uniform', clamping=False):
        assert output_distribution in ['uniform', 'normal'], \
            "output_distribution must be 'uniform' or 'normal'"
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.quantiles_ = None
        self.references_ = None
        self.n_quantiles_ = None
        self.clamping = False

    def _to_tensor(self, X):
        """Преобразует входной массив в torch.Tensor, если он не является таковым."""
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        return X

    def fit(self, X, y=None):
        """
        Вычисляет квантильные границы для каждого признака.
        
        X: array-like или torch.Tensor, shape (n_samples, n_features)
        """
        X = self._to_tensor(X)
        n_samples, n_features = X.shape
        # n_quant = min(self.n_quantiles, n_samples)
        n_quant = self.n_quantiles
        self.n_quantiles_ = n_quant
        # Опорные квантильные точки от 0 до 1
        print("n_quant", n_quant)
        self.references_ = torch.linspace(0, 1, n_quant, device=X.device, dtype=X.dtype)
        # Вычисляем квантильные границы по оси 0 (по образцам) для каждого признака.
        # Получаем тензор формы (n_quant, n_features)
        self.quantiles_ = torch.quantile(X, self.references_, dim=0)
        return self

    def transform(self, X):
        """
        Преобразует данные, используя вычисленные квантильные границы.
        
        X: array-like или torch.Tensor, shape (n_samples, n_features)
        
        Возвращает:
            torch.Tensor размера (n_samples, n_features). Если output_distribution='uniform',
            значения лежат в [0,1]; если 'normal' – распределены примерно по стандартному нормальному.
        """
        X = self._to_tensor(X)
        # Для каждого признака нам нужно выполнить поиск по квантильным границам.
        # Наши квантильные границы имеют форму (n_quant, n_features).
        # Для того чтобы для каждого признака (колонки) вызвать torch.searchsorted,
        # транспонируем данные так, чтобы:
        # - boundaries: shape = (n_features, n_quant)
        # - input: shape = (n_features, n_samples)
        boundaries = self.quantiles_.T           # shape: (n_features, n_quant)
        X_t = X.T                                # shape: (n_features, n_samples)
        # Теперь для каждого признака (каждая строка) boundaries[i] — 1D массив квантилей,
        # а X_t[i] — 1D массив значений. Вызываем searchsorted по строкам:
        indices = torch.searchsorted(boundaries, X_t, right=True)  # shape: (n_features, n_samples)
        # Приводим индексы к типу float и масштабируем в [0, 1]:
        y = indices.float() / (self.n_quantiles_ - 1)
        # Транспонируем результат обратно в форму (n_samples, n_features)
        y = y.T
        if self.clamping:
            y = torch.clamp(y, 0, 1)
        
        if self.output_distribution == 'normal':
            # Преобразуем равномерное распределение в нормальное:
            # Сначала масштабируем [0,1] в [-1,1]:
            y = 2 * y - 1
            # Чтобы избежать передачи точных значений -1 или 1 в erfinv, ограничим их:
            eps = 1e-6
            y = y.clamp(-1 + eps, 1 - eps)
            y = torch.erfinv(y) * np.sqrt(2)
        
        return y

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

class UpperQuartileNormalizer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for upper-quartile normalization.
    Normalizes each sample (row) by dividing its values by the 75th percentile.
    Supports numpy.ndarray, torch.Tensor, and pandas.DataFrame inputs.
    """
    
    def __init__(self, quantile: int = 75):
        self.quantile = quantile 
        
    def fit(self, X, y=None):
        """
        Fit method (not used, as no learning is needed for normalization).
        """
        return self

    def transform(self, X):
        """
        Apply upper-quartile normalization to the input data.

        Args:
            X (numpy.ndarray, torch.Tensor, or pandas.DataFrame): Input data array where rows are samples and columns are features.

        Returns:
            Same type as input: normalized data.
        """

        if isinstance(X, torch.Tensor):
            # Use torch operations
            q = torch.quantile(X, self.quantile / 100.0, dim=1, keepdim=True)
            q = torch.where(q == 0, torch.ones_like(q), q)
            normalized = X / q
            return normalized

        elif isinstance(X, np.ndarray):
            # Use numpy operations
            upper_quartiles = np.percentile(X, self.quantile, axis=1, keepdims=True)
            upper_quartiles[upper_quartiles == 0] = 1
            normalized = X / upper_quartiles
            return normalized

        elif isinstance(X, pd.DataFrame):
            # Operate on the underlying numpy array
            values = X.values
            upper_quartiles = np.percentile(values, self.quantile, axis=1, keepdims=True)
            upper_quartiles[upper_quartiles == 0] = 1
            normalized_values = values / upper_quartiles
            # Return DataFrame preserving columns and index
            return pd.DataFrame(normalized_values, columns=X.columns, index=X.index)

        else:
             raise TypeError(f"Unsupported data type: {type(X)}. Expected numpy.ndarray, torch.Tensor, or pandas.DataFrame.")
    


def log_transform(x):
    """
    Apply log(x + 1) transform.
    Supports numpy.ndarray, torch.Tensor, and pandas.DataFrame inputs.

    Args:
        x (numpy.ndarray, torch.Tensor, or pandas.DataFrame): Input array, tensor, or dataframe.

    Returns:
        Same type as input: transformed data.
    """
    if isinstance(x, torch.Tensor):
        return torch.log(x + 1)
    elif isinstance(x, np.ndarray):
        return np.log(x + 1)
    elif isinstance(x, pd.DataFrame):
        transformed_values = np.log(x.values + 1)
        return pd.DataFrame(transformed_values, columns=x.columns, index=x.index)
    else:
        raise TypeError(f"Unsupported data type: {type(x)}. Expected numpy.ndarray, torch.Tensor, or pandas.DataFrame.")

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
        if isinstance(sample, np.ndarray):
            return torch.from_numpy(sample)
        else:
            return sample

class Padding(object):
    def __init__(self, size):
        self.size = size
        

    def __call__(self, sample):
        padded_sample = torch.zeros((sample.shape[0],self.size))
        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample)
        padded_sample[:, :sample.shape[1]] = sample

        return padded_sample
    
def base_transforms(standart_scaler):
    return transforms.Compose([
                               transforms.Lambda(log_transform), 
                               Scale(standart_scaler)])
    
def padded_transforms(standart_scaler, size):
    return transforms.Compose([
                               transforms.Lambda(log_transform), 
                               Scale(standart_scaler),
                               Padding(size)])
    
def padded_transforms_simple(size):
    return  Padding(size)


def padded_transforms_with_scaling(standart_scaler2, size):
    if standart_scaler2 is not None:
        return transforms.Compose([
                                transforms.Lambda(log_transform),
                                Scale(UpperQuartileNormalizer(quantile=50)),
                                Scale(standart_scaler2),
                                Padding(size)
                                ])
    else:
        return transforms.Compose([
                                transforms.Lambda(log_transform),
                                Scale(UpperQuartileNormalizer(quantile=50)),
                                Padding(size)
                                ])
    
