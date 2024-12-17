import numpy as np
from torchvision import transforms

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
    
