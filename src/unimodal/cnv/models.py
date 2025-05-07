import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from src.unimodal.rna.mae import RnaMAEForPreTraining, RnaSurvivalModel, RnaMAEModel
from omegaconf import DictConfig

class CNVMAEModel(RnaMAEModel):
    def __init__(self, config):
        super().__init__(config)

class CNVMAEForPreTraining(RnaMAEForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        
class CNVSurvivalModel(RnaSurvivalModel):
    def __init__(self, config):
        super().__init__(config)
 
            
def initialise_cnv_mae_model(cfg):
    return CNVSurvivalModel(cfg)
