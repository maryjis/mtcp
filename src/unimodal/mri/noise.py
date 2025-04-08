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

"""
This classes adopted from  https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit_mae/modeling_vit_mae.py#L323
"""

class MriNoiseEmbeddings(MriMAEEmbeddings):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def random_noising(self, sequence, noise=None):
        """
        Perform per-sample random noising by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.noise_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # noise the second subset
        ids_noised = ids_shuffle[:, len_keep:]
        sequence_noised = torch.gather(sequence, dim=1, index=ids_noised.unsqueeze(-1).repeat(1, 1, dim))
        noise = torch.randn_like(sequence_noised)
        # noise_stds[noise_stds == 0] = 1 #very rare, but might be very painful
        if self.config.to_dict().get("noise_like_sample", False):
            means = sequence_noised.mean(dim=-1, keepdim=True)
            stds = sequence_noised.std(dim=-1, keepdim=True)
            noise_means = noise.mean(dim=-1, keepdim=True)
            noise_stds = noise.std(dim=-1, keepdim=True)
            noise = (noise - noise_means) / noise_stds * stds + means
        sequence_noised = sequence_noised * (1 - self.config.noise_power) + noise * self.config.noise_power

        sequence_processed = torch.cat([sequence_unmasked, sequence_noised], dim=1) #we return shuffled sequence due to torch.gather in decoder

        # generate the binary mask: 0 is keep, 1 is noised
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_processed, mask, ids_restore

    def forward(self, mri_values, noise=None, interpolate_pos_encoding: bool = False):
        batch_size, num_channels, mri_size1, mri_size2, mri_size3 = mri_values.shape
        embeddings = self.patch_embeddings(mri_values)
        
        # add position embeddings w/o cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]
     

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore = self.random_noising(embeddings, noise)

        # append cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore

class MriNoiseModel(MriMAEModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = MriNoiseEmbeddings(config)
        self.post_init()

class MriNoiseDecoder(MriMAEDecoder):
    def __init__(self, config, num_patches):
        super().__init__(config, num_patches)

    def forward(
        self,
        hidden_states,
        ids_restore,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        interpolate_pos_encoding: bool = False,
    ):
        # embed tokens
        x = self.decoder_embed(hidden_states)

        # append mask tokens to sequence
        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = x[:, 1:, :]  # no cls token
        # unshuffle
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]).to(x_.device))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # add pos embed
        if interpolate_pos_encoding:
            decoder_pos_embed = self.interpolate_pos_encoding(x)
        else:
            decoder_pos_embed = self.decoder_pos_embed
        hidden_states = x + decoder_pos_embed

        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    None,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)

        # remove cls token
        logits = logits[:, 1:, :]

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        return ViTMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class MriNoiseForPreTraining(MriMAEForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.vit = MriNoiseModel(config)
        self.decoder = MriNoiseDecoder(config, num_patches=self.vit.embeddings.num_patches)

        # Initialize weights and apply final processing
        self.post_init()

class MriNoiseSurvivalModel(MriMaeSurvivalModel):
    def __init__(self, config):
        super().__init__(config)
        if config.to_dict().get("is_load_pretrained", False):
            self.vit = MriNoiseModel.from_pretrained(config.pretrained_model_path, config = config)
            print(f"Pretrained model loaded from {config.pretrained_model_path}")
        else:
            self.vit = MriNoiseModel(config)
