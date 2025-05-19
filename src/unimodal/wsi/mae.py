import torch
import torch.nn as nn
from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEModel,
    ViTMAEDecoder,
    ViTMAEForPreTraining,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed
)
from einops import rearrange, reduce
import numpy as np
import os
import math
from math import ceil
from omegaconf import OmegaConf

# Import necessary components from utils.py
from ...utils import (
    exists,
    NystromAttention,
    TransLayer,
    PPEG
)


class WsiMAEPatchEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.image_size = cfg.image_size
        self.patch_size = cfg.patch_size
        self.hidden_size = cfg.hidden_size
        self.num_channels = cfg.num_channels
        
        max_patches_per_sample = cfg.max_patches_per_sample
        self.random_patch_selection = cfg.random_patch_selection
        
        # Number of subpatches in a single image of size image_size x image_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
       
        if self.random_patch_selection is True:
            # Total number of patches after splitting all max_patches_per_sample
            self.total_patches = self.num_patches
        else:
            self.total_patches = max_patches_per_sample * self.num_patches

        if cfg.to_dict().get("embeddings_layers", None):
            c = OmegaConf.create(cfg.to_dict())
            self.projection = nn.Sequential(*[getattr(nn, layer["name"])(*layer.get("args", []), **layer.get("kwargs", {})) for layer in c["embeddings_layers"]])
        else:
            self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def _split_patches(self, wsi_tensor):
        """
        Splits the input patches into subpatches.
        Expected input: [batch_size, max_patches_per_sample, 3, 256, 256]
        """
        b, n, c, h, w = wsi_tensor.shape
        p = self.patch_size

        if h % p != 0 or w % p != 0:
            raise ValueError(f"Image size ({h}, {w}) is not evenly divisible by ({p}, {p}).")

        # unfold + permute + view
        patches = wsi_tensor.unfold(3, p, p).unfold(4, p, p)  # -> [b, n, c, h//p, p, w//p, p]
        patches = patches.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        patches = patches.view(b, n * self.num_patches, c, p, p)

        return patches

    def forward(self, wsi_tensor):
        # Splitting into subpatches
        # patches = self._split_patches(wsi_tensor)  
        patches = wsi_tensor
        b, n, c, h, w = patches.shape
        # Reshape to (batch_size * num_patches, c, h, w)
        patches = rearrange(patches, 'b n c h w -> (b n) c h w')

        x = self.projection(patches).flatten(2).transpose(1, 2)  # [b*n, hidden_size]


        return x

class WsiMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position, and patch embeddings for 2D data (e.g., images).
    """
    def __init__(self, cfg):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.patch_embeddings = WsiMAEPatchEmbeddings(cfg)
        self.num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.patch_embeddings.num_patches + 1, cfg.hidden_size), requires_grad=False
        )
        self.patch_size = cfg.patch_size
        self.config = cfg
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1], int(self.patch_embeddings.num_patches**0.5), add_cls_token=True
        )
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        if isinstance(self.patch_embeddings.projection, nn.Sequential):
            # Initialize the last layer in Sequential, if it's Conv2d or Linear
            for m in self.patch_embeddings.projection.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    w = m.weight.data
                    torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        else:
            # projection - 1 layer
            w = self.patch_embeddings.projection.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)
        

    def random_masking(self, sequence, noise=None):
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)
            
        if self.config.mask_ratio>0:
            ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)
            ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)
        else:
            ids_shuffle = torch.arange(seq_length).repeat(batch_size, 1).to(sequence.device)
            ids_restore = torch.arange(seq_length).repeat(batch_size, 1).to(sequence.device)

            
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, 1, ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return sequence_unmasked, mask, ids_restore

    def forward(self, image_values, noise=None, interpolate_pos_encoding: bool = False):
        batch_size, num_patches, num_channels, img_height, img_width = image_values.shape
        embeddings = self.patch_embeddings(image_values)
        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)
        
        # CLS token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore


class WsiMAEModel(ViTMAEModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = WsiMAEEmbeddings(config)
        self.post_init()

    def patchify(self, imgs, interpolate_pos_encoding: bool = False):
        p = self.config.patch_size
        b, n, c, h, w = imgs.shape
        assert h == w == self.config.image_size 
        assert h % p == 0 and w % p == 0 
        patches = rearrange(imgs, 'b n c (h p1) (w p2) -> (b n) (h w) (c p1 p2)', p1=p, p2=p)
        # patches = patches.flatten(2)
        return patches
    
    def unpatchify(self, imgs):
        batch_size = imgs.shape[0]
        patch_size = self.config.patch_size
        channels = self.config.num_channels
        image_size = self.config.image_size
        
        h = w = image_size // patch_size

        return rearrange(
            imgs,
            'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
            h=h,
            w=w,
            c=channels,
            p1=patch_size,
            p2=patch_size
        )

    def forward(self, imgs, is_multimodal: bool = False, **kwargs):
        print(f"[WsiMAEModel.forward] Input imgs shape: {imgs.shape}")
        out = super().forward(imgs)
        print(f"[WsiMAEModel.forward] After super().forward - last_hidden_state shape: {out.last_hidden_state.shape}")
        
        if not self.config.random_patch_selection:
            N = self.config.max_patches_per_sample
            B_new = out.last_hidden_state.shape[0]
            B = B_new // N
            seq_length = out.last_hidden_state.shape[1]
            print(f"[WsiMAEModel.forward] N={N}, B_new={B_new}, B={B}, seq_length={seq_length}")
            
            out.last_hidden_state = out.last_hidden_state.view(B, N, seq_length, -1).mean(dim=1)
            out.mask = out.mask.view(B, N, -1).mean(dim=1)
            out.ids_restore = torch.arange(seq_length-1, device=out.ids_restore.device).repeat(B, 1)
            print(f"[WsiMAEModel.forward] After reshape - last_hidden_state shape: {out.last_hidden_state.shape}")
        return out


class WsiMAEDecoderPred(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.image_size % config.patch_size == 0
        self.num_patches_per_dim = config.image_size // config.patch_size
        if config.to_dict().get("decoder_pred_layers", None): 
            c = OmegaConf.create(config.to_dict())
            self.projector = nn.Sequential(*[getattr(nn, layer["name"])(*layer.get("args", []), **layer.get("kwargs", {})) for layer in c["decoder_pred_layers"]])
        else:        
            self.projector = nn.ConvTranspose2d(
                config.decoder_hidden_size,
                config.num_channels,
                config.patch_size,
                stride=config.patch_size)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(-1).unsqueeze(-1) 
        x = rearrange(x, 'b s e x y -> (b s) e x y')
        x = self.projector(x)
        x = rearrange(x, '(b s) c h w -> b s (c h w)', b=batch_size)
        return x


class WsiMAEDecoder(ViTMAEDecoder):
    def __init__(self, config, num_patches):
        super().__init__(config, num_patches)
        self.decoder_pred = WsiMAEDecoderPred(config)
        self.initialize_weights(num_patches)
        
    def initialize_weights(self, num_patches):
        decoder_pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.decoder_pos_embed.shape[-1], np.arange(int(num_patches), dtype=np.float32))
        decoder_pos_embed = np.concatenate([
            np.zeros([1, self.decoder_pos_embed.shape[-1]]),
            decoder_pos_embed
        ], axis=0)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)


class WsiMAEForPreTraining(ViTMAEForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vit = WsiMAEModel(config)
        self.decoder = WsiMAEDecoder(config, num_patches=self.vit.embeddings.num_patches)
        self.post_init()
        
    def patchify(self, imgs, interpolate_pos_encoding: bool = False):
        p = self.config.patch_size
        b, n, c, h, w = imgs.shape
        assert h == w == self.config.image_size 
        assert h % p == 0 and w % p == 0
        patches = rearrange(imgs, 'b n c (h p1) (w p2) -> (b n) (h w) (c p1 p2)', p1=p, p2=p)
        # patches = patches.flatten(2)
        return patches


class WsiMaeSurvivalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.to_dict().get("is_load_pretrained", False):
            self.vit = WsiMAEModel.from_pretrained(config.pretrained_model_path, config=config)
            print(f"Pretrained model loaded from {config.pretrained_model_path}")
        else:
            self.vit = WsiMAEModel(config)
        self.projection = nn.Linear(config.hidden_size, config.output_dim)
        self.max_patches_per_sample = config.max_patches_per_sample
        self.use_transformer_pool = config.use_transformer_pool
        
        # Add components for transformer pool
        if self.use_transformer_pool:
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
            nn.init.normal_(self.cls_token, std=1e-6)
            self.trans_layer1 = TransLayer(dim=config.hidden_size)
            self.pos_layer = PPEG(dim=config.hidden_size)
            self.trans_layer2 = TransLayer(dim=config.hidden_size)
            self.norm = nn.LayerNorm(config.hidden_size)
            
    def forward(self, wsi_values, masks=None):
        print(f"[WsiMaeSurvivalModel.forward] Input wsi_values shape: {wsi_values.shape}")
        vit_out = self.vit(wsi_values)
        print(f"[WsiMaeSurvivalModel.forward] vit_out.last_hidden_state shape: {vit_out.last_hidden_state.shape}")
        
        cls_tokens = vit_out.last_hidden_state[:, 0, :]  # [B_new, hidden_size]
        print(f"[WsiMaeSurvivalModel.forward] cls_tokens shape after extraction: {cls_tokens.shape}")
        
        if self.use_transformer_pool:
            # Reshape classification tokens and get patient representation
            features = cls_tokens.unsqueeze(1)  # [B_new, 1, hidden_size]
            print(f"[WsiMaeSurvivalModel.forward] features shape after unsqueeze: {features.shape}")
            
            # 1. Padding to the nearest square number of tokens
            H = features.shape[1]
            _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
            add_length = _H * _W - H
            if add_length > 0:
                h = torch.cat([features, features[:, :add_length, :]], dim=1)
            else:
                h = features
                
            # 2. Append cls_token at the beginning
            cls_tokens_pool = self.cls_token.expand(features.shape[0], -1, -1).to(h.device)
            h = torch.cat((cls_tokens_pool, h), dim=1)
            
            # 3. First TransLayer
            h = self.trans_layer1(h)
            
            # 4. PPEG
            h = self.pos_layer(h, _H, _W)
            
            # 5. Second TransLayer
            h = self.trans_layer2(h)
            
            # 6. Final LayerNorm
            h = self.norm(h)
            
            # 7. Output - first token as patient representation
            patient_repr = h[:, 0]
            x = self.projection(patient_repr)
        else:
            print(f"[WsiMaeSurvivalModel.forward] Before projection - cls_tokens shape: {cls_tokens.shape}")
            x = self.projection(cls_tokens)
            print(f"[WsiMaeSurvivalModel.forward] Final output shape: {x.shape}")
            
        return x.squeeze(-1)
