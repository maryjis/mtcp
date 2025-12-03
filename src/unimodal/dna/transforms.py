from src.unimodal.rna.transforms import Padding
from torchvision.transforms import ToTensor
from torch.nn import Identity
import numpy as np
from torchvision import transforms
from sklearn.base import BaseEstimator, TransformerMixin
import torch 

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
        
    
def padded_transforms_simple(size):
    if size:
        return  Padding(size)
    else:
        return None
    
    
def padded_transforms_scaling(standart_scaler, size=None, is_padding =True):
    print("padded_transforms_scaling: ", size)
    print("is_padding: ", is_padding)
    if standart_scaler is not None:
        if is_padding:
            return transforms.Compose([
                                    Scale(standart_scaler),
                                    Padding(size)])
        else:
            return transforms.Compose([
                                    Scale(standart_scaler)])                            
    else:
        if is_padding:
            return Padding(size)
        else:
            return Identity()