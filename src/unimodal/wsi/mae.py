import torch
import torch.nn as nn
from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEModel,
    ViTMAEDecoder,
    ViTMAEForPreTraining,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed
)
from einops import rearrange
import numpy as np
import os


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

        # Pass through several small convolutional layers + pooling
        x = self.projection(patches).flatten(2).transpose(1, 2)  # [b*n, hidden_size]
        # Return to (batch_size, num_patches, hidden_size)
        # x = rearrange(x, '(b n) c -> b n c', b=b, n=self.total_patches)

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
    
    def unpatchify(self, imgs, original_size: int =None):
        batch_size = imgs.shape[0]
        return imgs.reshape(batch_size,
                            -1,
                            self.config.num_channels,
                            self.config.image_size,
                            self.config.image_size)

    def forward(self, imgs, is_multimodal: bool = False):
        out = super().forward(imgs)
        if is_multimodal and not self.config.random_patch_selection:
            N = self.config.max_patches_per_sample
            B_new = out.last_hidden_state.shape[0]
            B = B_new // N
            seq_length = out.last_hidden_state.shape[1]
            out.last_hidden_state = out.last_hidden_state.view(B, N, seq_length, -1).mean(dim=1)
            out.mask = out.mask.view(B, N, -1).mean(dim=1)
            out.ids_restore =torch.arange(seq_length-1, device=out.ids_restore.device).repeat(B, 1)   
        return out


class WsiMAEDecoderPred(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.image_size % config.patch_size == 0
        self.num_patches_per_dim = config.image_size // config.patch_size
        self.projector = nn.ConvTranspose2d(
            config.decoder_hidden_size,
            config.num_channels,
            config.patch_size,
            stride=config.patch_size
        )

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

    def forward(self, wsi_values, masks=None):
        vit_out = self.vit(wsi_values)
        cls_tokens = vit_out.last_hidden_state[:, 0, :]  # [B_new, hidden_size]
        N = self.max_patches_per_sample
        B_new = cls_tokens.shape[0]
        B = B_new // N
        cls_tokens = cls_tokens.view(B, N, -1)  # [B, N, hidden_size]
        patient_repr = cls_tokens.mean(dim=1)  # [B, hidden_size]
        x = self.projection(patient_repr)
        return x.squeeze(-1)
