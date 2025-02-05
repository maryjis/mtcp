import torch.nn as nn
import torch
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModel,ViTMAEDecoder, ViTMAEForPreTraining, get_1d_sincos_pos_embed_from_grid
import numpy as np
from typing import Optional, Set, Tuple, Union

"""
This classes adopted from  https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit_mae/modeling_vit_mae.py#L323
"""

class RnaMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, cfg):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.patch_embeddings = RnaTMAEPatchEmbeddings(cfg)
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, cfg.hidden_size), requires_grad=False
        )
        self.patch_size = cfg.patch_size
        self.config = cfg
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.position_embeddings.shape[-1], 
            np.arange(int(self.patch_embeddings.num_patches), dtype=np.float32)     )
        pos_embed = np.concatenate([np.zeros([1, self.position_embeddings.shape[-1]]), pos_embed], axis=0)
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)


    def random_masking(self, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def forward(self, rna_values, noise=None, interpolate_pos_encoding: bool = False):
        batch_size, num_channels, rna_size = rna_values.shape
        embeddings = self.patch_embeddings(rna_values)
        
        # add position embeddings w/o cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]
     

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        # append cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore


class RnaTMAEPatchEmbeddings(nn.Module):
    """
    This class turns `rna_values` of shape `(batch_size, 1, rna_size)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, cfg):
        super().__init__()
        rna_size, patch_size = cfg.size, cfg.patch_size
        num_channels, hidden_size = cfg.num_channels, cfg.hidden_size

        num_patches = rna_size // patch_size
        self.rna_size = rna_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv1d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, rna_values):
        
        batch_size, num_channels, rna_size = rna_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the rna values match with the one set in the configuration."
            )
        #todo add padding OR interpolate_pos_encoding ?    
        x = self.projection(rna_values).transpose(1, 2) 
        return x


class RnaMAEModel(ViTMAEModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = RnaMAEEmbeddings(config)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.post_init()
        
    def patchify(self, rna_values, interpolate_pos_encoding: bool = False):
        """
        Args:
            rna_values (`torch.FloatTensor` of shape `(batch_size, num_channels, rna_size)`):
                RNA values.
            interpolate_pos_encoding (`bool`, *optional*, default `False`):
                interpolation flag passed during the forward pass.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size * num_channels)`:
                Patchified pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        # sanity checks
        if not interpolate_pos_encoding and (
            rna_values.shape[2] % patch_size != 0
        ):
            raise ValueError("Make sure the RNA values is divisible by the patch size")
        if rna_values.shape[1] != num_channels:
            raise ValueError(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            )

        # patchify
        batch_size = rna_values.shape[0]
        num_patches = rna_values.shape[2] // patch_size

        patchified_rna_values = rna_values.reshape(batch_size, num_channels, num_patches, patch_size)
        patchified_rna_values = patchified_rna_values.permute(0, 2, 3, 1)
        patchified_rna_values = patchified_rna_values.reshape(
            batch_size, num_patches, patch_size * num_channels
        )
        return patchified_rna_values

    def unpatchify(self, patchified_rna_values, original_rna_size: int=None):
        """
        Args:
            patchified_rna_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size * num_channels)`:
                Patchified rna values.
            original_image_size (`int`, *optional*):
                Original rna size.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, original_rna_size)`:
                Pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        original_rna_size = original_rna_size if original_rna_size is not None else self.config.size
        
        num_patches = original_rna_size // patch_size
   
        # sanity check
        if num_patches != patchified_rna_values.shape[1]:
            raise ValueError(
                f"The number of patches in the patchified rna values {patchified_rna_values.shape[1]}, does not match the number of patches on original rna {num_patches}"
            )

        # unpatchify
        batch_size = patchified_rna_values.shape[0]
        patchified_pixel_values = patchified_rna_values.reshape(
            batch_size,
            num_patches,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = patchified_pixel_values.permute(0, 3, 1, 2)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches * patch_size
        )
        return pixel_values
    
        

class RnaMAEDecoder(ViTMAEDecoder):
    def __init__(self, config, num_patches):
        super().__init__(config, num_patches)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size * config.num_channels, bias=True
        )  # encoder to decoder
        self.initialize_weights(num_patches)
        
    def initialize_weights(self, num_patches):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.decoder_pos_embed.shape[-1], np.arange(int(num_patches), dtype=np.float32))
        decoder_pos_embed = np.concatenate([np.zeros([1, self.decoder_pos_embed.shape[-1]]), decoder_pos_embed], axis=0)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)


class RnaMAEForPreTraining(ViTMAEForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.vit = RnaMAEModel(config)
        self.decoder = RnaMAEDecoder(config, num_patches=self.vit.embeddings.num_patches)

        # Initialize weights and apply final processing
        self.post_init()
        
    def patchify(self, rna_values, interpolate_pos_encoding: bool = False):
        """
        Args:
            rna_values (`torch.FloatTensor` of shape `(batch_size, num_channels, rna_size)`):
                RNA values.
            interpolate_pos_encoding (`bool`, *optional*, default `False`):
                interpolation flag passed during the forward pass.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size * num_channels)`:
                Patchified pixel values.
        """
        # TODO return self.vit.patchify(self, rna_values, interpolate_pos_encoding: bool = False)
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        # sanity checks
        if not interpolate_pos_encoding and (
            rna_values.shape[2] % patch_size != 0
        ):
            raise ValueError("Make sure the RNA values is divisible by the patch size")
        if rna_values.shape[1] != num_channels:
            raise ValueError(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            )

        # patchify
        batch_size = rna_values.shape[0]
        num_patches = rna_values.shape[2] // patch_size

        patchified_rna_values = rna_values.reshape(batch_size, num_channels, num_patches, patch_size)
        patchified_rna_values = patchified_rna_values.permute(0, 2, 3, 1)
        patchified_rna_values = patchified_rna_values.reshape(
            batch_size, num_patches, patch_size * num_channels
        )
        return patchified_rna_values

    def unpatchify(self, patchified_rna_values, original_rna_size: int):
        """
        Args:
            patchified_rna_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size * num_channels)`:
                Patchified rna values.
            original_image_size (`int`, *optional*):
                Original rna size.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, original_rna_size)`:
                Pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        original_rna_size = original_rna_size if original_rna_size is not None else self.config.rna_size
        
        num_patches = original_rna_size // patch_size
   
        # sanity check
        if num_patches != patchified_rna_values.shape[1]:
            raise ValueError(
                f"The number of patches in the patchified rna values {patchified_rna_values.shape[1]}, does not match the number of patches on original rna {num_patches}"
            )

        # unpatchify
        batch_size = patchified_rna_values.shape[0]
        patchified_pixel_values = patchified_rna_values.reshape(
            batch_size,
            num_patches,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = patchified_pixel_values.permute(0, 3, 1, 2)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches * patch_size
        )
        return pixel_values


class RnaSurvivalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config =config
        if config.is_load_pretrained:
            self.vit = RnaMAEModel.from_pretrained(self.config.pretrained_model_path, config = self.config)
        else:
            self.vit = RnaMAEModel(config)
        self.projection = nn.Linear(self.config.hidden_size, self.config.output_dim)
        
    def forward(self, rna_values, masks=None):
        x = self.vit(rna_values)
        x = self.projection(x.last_hidden_state[:,0,:])
        return x.squeeze(-1)
    
    
    
def initialise_rna_mae_model(cfg):
    return RnaSurvivalModel(cfg)