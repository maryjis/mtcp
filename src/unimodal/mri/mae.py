from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEForPreTraining, 
    ViTMAEModel, 
    get_1d_sincos_pos_embed_from_grid,
    ViTMAEConfig,
    ViTMAEDecoder
)
import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
from omegaconf import OmegaConf

class MriTMAEPatchEmbeddings(nn.Module):
    """
    This class turns `mri_values` of shape `(batch_size, channels, mri_size, mri_size, mri_size)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, cfg):
        super().__init__()
        mri_size, patch_size = cfg.mri_size, cfg.patch_size
        num_channels, hidden_size = cfg.num_channels, cfg.hidden_size
        assert mri_size%patch_size==0, "Mri size must be divisible by patch size"
        num_patches = mri_size**3 // patch_size**3

        self.mri_size = mri_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        if cfg.to_dict().get("embeddings_layers", None):
            c = OmegaConf.create(cfg.to_dict())
            self.projection = nn.Sequential(*[getattr(nn, layer["name"])(*layer.get("args", []), **layer.get("kwargs", {})) for layer in c["embeddings_layers"]])
        else:
            print("WARNING: you haven't parametrized encoder embeddings layers in model config, heavy default convolution is used")
            self.projection = nn.Conv3d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, mri_values):
        batch_size, num_channels, mri_size1, mri_size2, mri_size3 = mri_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the mri values match with the one set in the configuration."
            )
          
        mri_values = rearrange(
            mri_values, 
            'b c (x p_x) (y p_y) (z p_z) -> (b x y z) c p_x p_y p_z', 
            p_x=self.patch_size, 
            p_y=self.patch_size, 
            p_z=self.patch_size
        )
        x = self.projection(mri_values) # B*S,C,P_x,P_y,P_z -> B*S,E,1,1,1
        x = rearrange(x.squeeze(), '(b s) e -> b s e', b=batch_size)
        return x

"""
This classes adopted from  https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit_mae/modeling_vit_mae.py#L323
"""

class MriMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, cfg):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.patch_embeddings = MriTMAEPatchEmbeddings(cfg)
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
            np.arange(int(self.patch_embeddings.num_patches), dtype=np.float32)
        )
        pos_embed = np.concatenate([
            np.zeros([1, self.position_embeddings.shape[-1]]), #for cls token
            pos_embed
        ], axis=0)
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        for layer in self.patch_embeddings.projection.modules():
            if hasattr(layer, "weight"):
                w = layer.weight.data
                torch.nn.init.xavier_uniform_(w)

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

    def forward(self, mri_values, noise=None, interpolate_pos_encoding: bool = False):
        batch_size, num_channels, mri_size1, mri_size2, mri_size3 = mri_values.shape
        embeddings = self.patch_embeddings(mri_values)
        
        # add position embeddings w/o cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]
     

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        # append cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore

class MriMAEModel(ViTMAEModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = MriMAEEmbeddings(config)
        self.post_init()
        
    def patchify(self, imgs, interpolate_pos_encoding: bool = False):
        return rearrange(
            imgs, 
            'b c (h p1) (w p2) (d p3) -> b (h w d) (c p1 p2 p3)', #for masking input should be in format (B, S, E) 
            p1=self.config.patch_size, 
            p2=self.config.patch_size, 
            p3=self.config.patch_size
        )
        
    def unpatchify(self, imgs, original_size: int =None):
        batch_size = imgs.shape[0]
        return rearrange(
            imgs, 
            'b (h w d) (c p1 p2 p3) -> b c (h p1) (w p2) (d p3)',
            c = 1,
            h = 4,
            w = 4,
            d = 4,
            p1=self.config.patch_size, 
            p2=self.config.patch_size, 
            p3=self.config.patch_size
        )
        
        # return imgs.reshape(batch_size,
        #                     self.config.num_channels,
        #                     self.config.mri_size,
        #                     self.config.mri_size,
        #                     self.config.mri_size)

class MriMAEDecoderPred(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.mri_size % config.patch_size == 0, "MRI size must be divisible by patch size"
        self.num_patches_along_axis = config.mri_size // config.patch_size
        
        if config.to_dict().get("decoder_pred_layers", None): 
            c = OmegaConf.create(config.to_dict())
            self.projector = nn.Sequential(*[getattr(nn, layer["name"])(*layer.get("args", []), **layer.get("kwargs", {})) for layer in c["decoder_pred_layers"]])
        else:
            print("WARNING: you haven't parametrized decoder projector layers in model config, heavy default convolution is used")
            self.projector = nn.ConvTranspose3d(
                config.decoder_hidden_size, 
                config.num_channels, 
                config.patch_size, 
                stride=config.patch_size
            )

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1) # [B, S, E] -> [B, S, E, 1, 1, 1]
        x = rearrange(x, 'b s e x y z -> (b s) e x y z')
        x = self.projector(x) # [B*S, E, 1, 1, 1] -> [(B*S), C, 16, 16, 16]
        x = rearrange(x, '(b s) c x y z -> b s (c x y z)', b=batch_size)
        return x

class MriMAEDecoder(ViTMAEDecoder):
    def __init__(self, config, num_patches):
        super().__init__(config, num_patches)
        self.decoder_pred = MriMAEDecoderPred(config)
        self.initialize_weights(num_patches)
        
    def initialize_weights(self, num_patches):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.decoder_pos_embed.shape[-1], np.arange(int(num_patches), dtype=np.float32))
        decoder_pos_embed = np.concatenate([
            np.zeros([1, self.decoder_pos_embed.shape[-1]]), 
            decoder_pos_embed
        ], axis=0)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

class MriMAEForPreTraining(ViTMAEForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.vit = MriMAEModel(config)
        self.decoder = MriMAEDecoder(config, num_patches=self.vit.embeddings.num_patches)

        # Initialize weights and apply final processing
        self.post_init()

    def patchify(self, imgs, interpolate_pos_encoding: bool = False):
        return rearrange(
            imgs, 
            'b c (h p1) (w p2) (d p3) -> b (h w d) (c p1 p2 p3)', #for masking input should be in format (B, S, E) 
            p1=self.config.patch_size, 
            p2=self.config.patch_size, 
            p3=self.config.patch_size
        )
    

class MriMaeSurvivalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.to_dict().get("is_load_pretrained", False):
            self.vit = MriMAEModel.from_pretrained(config.pretrained_model_path, config = config)
            print(f"Pretrained model loaded from {config.pretrained_model_path}")
        else:
            self.vit = MriMAEModel(config)
        self.projection = nn.Linear(config.hidden_size, config.output_dim)
        
    def forward(self, mri_values, masks=None):
        x = self.vit(mri_values)

        x = self.projection(x.last_hidden_state[:,0,:])
        return x.squeeze(-1)