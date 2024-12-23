import numpy as np
from torchvision import transforms
from sklearn.base import BaseEstimator, TransformerMixin


class UpperQuartileNormalizer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for upper-quartile normalization.
    Normalizes each sample (row) by dividing its values by the 75th percentile.
    """
    
    def __init__(self, quantile: int = 75):
        self.quantile = quantile 
        
    def fit(self, X, y=None):
        """
        Fit method (not used, as no learning is needed for normalization).
        """
        return self  # Nothing to fit, return self

    def transform(self, X):
        """
        Apply upper-quartile normalization to the input data.

        Args:
            X (numpy.ndarray): Input data array where rows are samples and columns are features.

        Returns:
            numpy.ndarray: Normalized data.
        """
        # Calculate the 75th percentile (upper quartile) for each sample (row)
        upper_quartiles = np.percentile(X, self.quantile, axis=1, keepdims=True)
        
        # Avoid division by zero
        upper_quartiles[upper_quartiles == 0] = 1
        
        # Normalize each value by dividing by the upper quartile
        return X / upper_quartiles
    
    
def log_transform(x):
    return np.log(x + 1)

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

class Padding(object):
    def __init__(self, size):
        self.size = size
        

    def __call__(self, sample):
        padded_sample = np.zeros((sample.shape[0],self.size))
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
    
