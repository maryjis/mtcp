from src.unimodal.rna.transforms import Padding
from torchvision.transforms import ToTensor
from torch.nn import Identity

def padded_transforms_simple(size):
    if size:
        return  Padding(size)
    else:
        return None