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

def base_transforms(standart_scaler):
    return transforms.Compose([
                               transforms.Lambda(log_transform), 
                               Scale(standart_scaler)])