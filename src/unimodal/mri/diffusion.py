from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEForPreTraining, 
    ViTMAEModel, 
    get_1d_sincos_pos_embed_from_grid,
    ViTMAEConfig,
    ViTMAEDecoder,
    ViTMAEForPreTrainingOutput,
    ViTMAEDecoderOutput
)
import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
from omegaconf import OmegaConf
from typing import Optional, Tuple, Union

from .mae import MriTMAEPatchEmbeddings, MriMAEEmbeddings, MriMAEModel, MriMAEDecoder, MriMAEForPreTraining, MriMaeSurvivalModel, MriMAEDecoderPred
from .noise import MriNoiseDecoder

"""
This classes adopted from  https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit_mae/modeling_vit_mae.py#L323
"""

class MriDiffusionEmbeddings(MriMAEEmbeddings):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def random_nothing(self, sequence):
        """
        Perform nothing
        """
        batch_size, seq_length, dim = sequence.shape
        idx_restore = torch.arange(seq_length, device=sequence.device).repeat(batch_size, 1)
        mask = torch.zeros([batch_size, seq_length], device=sequence.device)

        return sequence, mask, idx_restore

    def forward(self, mri_values, noise=None, interpolate_pos_encoding: bool = False):
        batch_size, num_channels, mri_size1, mri_size2, mri_size3 = mri_values.shape
        embeddings = self.patch_embeddings(mri_values)
        
        # add position embeddings w/o cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]
     
        embeddings, mask, ids_restore = self.random_nothing(embeddings)

        # append cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore

class MriDiffusionModel(MriMAEModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = MriDiffusionEmbeddings(config)
        self.post_init()

class MriDiffusionDecoder(MriNoiseDecoder):
    def __init__(self, config, num_patches):
        super().__init__(config, num_patches)

class MriDiffusionForPreTraining(MriMAEForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.vit = MriDiffusionModel(config)
        self.decoder = MriDiffusionDecoder(config, num_patches=self.vit.embeddings.num_patches)

        # Initialize weights and apply final processing
        self.post_init()

    def add_noise(self, x, alpha_t):
        noise = torch.randn_like(x)
        alpha_t = alpha_t.view(-1, *np.ones(x.ndim-1, dtype=int))  # <- B, C=1, H=1, W=1, D=1
        return x * (1 - alpha_t) + noise * alpha_t
    
    def forward_loss(self, pixel_values, pred, mask, interpolate_pos_encoding: bool = False):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.
            pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Predicted pixel values.
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).
            interpolate_pos_encoding (`bool`, *optional*, default `False`):
                interpolation flag passed during the forward pass.

        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """
        target = self.patchify(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = torch.nn.functional.mse_loss(pred, target)
        return loss

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple, ViTMAEForPreTrainingOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        alpha_t = torch.rand(pixel_values.shape[0]).to(pixel_values)    # Pick random noise amounts
        noisy_pixel_values = self.add_noise(pixel_values, alpha_t)      # Create our noisy x

        ################
        # models forward
        outputs = self.vit(
            noisy_pixel_values,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, ids_restore, interpolate_pos_encoding=interpolate_pos_encoding)
        logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        # models forward
        ################
        
        loss = self.forward_loss(pixel_values, logits, mask, interpolate_pos_encoding=interpolate_pos_encoding)

        if not return_dict:
            output = (logits, mask, ids_restore) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ViTMAEForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class MriDiffusionSurvivalModel(MriMaeSurvivalModel):
    def __init__(self, config):
        super().__init__(config)
        if config.to_dict().get("is_load_pretrained", False):
            self.vit = MriDiffusionModel.from_pretrained(config.pretrained_model_path, config = config)
            print(f"Pretrained model loaded from {config.pretrained_model_path}")
        else:
            self.vit = MriDiffusionModel(config)
