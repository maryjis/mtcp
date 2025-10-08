import torch
import torch.nn as nn
from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEModel,
    ViTMAEDecoder,
    ViTMAEForPreTraining,
    ViTMAEPatchEmbeddings,
    ViTMAEModelOutput, 
    get_2d_sincos_pos_embed)
from typing import Callable, Optional, Union
from einops import rearrange, reduce

class ViTMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ViTMAEPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size), requires_grad=False
        )
        self.patch_size = config.patch_size
        self.config = config

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

    # Copied from transformers.models.vit.modeling_vit.ViTEmbeddings.interpolate_pos_encoding
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

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
        print("self.config.mask_ratio:", self.config.mask_ratio)
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

    def forward(self, pixel_values, noise=None, interpolate_pos_encoding: bool = False):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        if interpolate_pos_encoding:
            position_embeddings = self.interpolate_pos_encoding(embeddings, height, width)
        else:
            position_embeddings = self.position_embeddings

        # add position embeddings w/o cls token
        embeddings = embeddings + position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        # append cls token
        cls_token = self.cls_token + position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore

class WsiMAEModel(ViTMAEModel):
    def __init__(self, config):
        print("WsiMAEModel: ", config)
        super().__init__(config)
        self.embeddings = ViTMAEEmbeddings(config)
        
    def unpatchify(self, patchified_pixel_values, original_image_size: Optional[tuple[int, int]] = None):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
            original_image_size (`tuple[int, int]`, *optional*):
                Original image size.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                Pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        original_image_size = (
            original_image_size
            if original_image_size is not None
            else (self.config.image_size, self.config.image_size)
        )
        original_height, original_width = original_image_size
        num_patches_h = original_height // patch_size
        num_patches_w = original_width // patch_size
        # sanity check
        if num_patches_h * num_patches_w != patchified_pixel_values.shape[1]:
            raise ValueError(
                f"The number of patches in the patchified pixel values {patchified_pixel_values.shape[1]}, does not match the number of patches on original image {num_patches_h}*{num_patches_w}"
            )

        # unpatchify
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_h,
            num_patches_w,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_h * patch_size,
            num_patches_w * patch_size,
        )
        return pixel_values
        
    def patchify(self, pixel_values, interpolate_pos_encoding: bool = False):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.
            interpolate_pos_encoding (`bool`, *optional*, default `False`):
                interpolation flag passed during the forward pass.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """

        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        # sanity checks
        if not interpolate_pos_encoding and (
            pixel_values.shape[2] != pixel_values.shape[3] or pixel_values.shape[2] % patch_size != 0
        ):
            raise ValueError("Make sure the pixel values have a squared size that is divisible by the patch size")
        if pixel_values.shape[1] != num_channels:
            raise ValueError(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            )

        # patchify
        batch_size = pixel_values.shape[0]
        num_patches_h = pixel_values.shape[2] // patch_size
        num_patches_w = pixel_values.shape[3] // patch_size
        patchified_pixel_values = pixel_values.reshape(
            batch_size, num_channels, num_patches_h, patch_size, num_patches_w, patch_size
        )
        patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size, num_patches_h * num_patches_w, patch_size**2 * num_channels
        )
        return patchified_pixel_values
        
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[tuple, ViTMAEModelOutput]:
        pixel_values = rearrange(pixel_values, 'b n c h w -> (b n) c h w')

        encoded = super().forward(pixel_values=pixel_values,
                                  noise=noise,
                                  head_mask=head_mask,
                                  output_attentions=output_attentions,
                                  output_hidden_states =output_hidden_states,
                                  return_dict=return_dict, 
                                  interpolate_pos_encoding=interpolate_pos_encoding)
        return encoded
    
class WsiMAEDecoder(ViTMAEDecoder):
    def __init__(self, config, num_patches):
        super().__init__(config, num_patches)
        
        
class WsiMAEForPreTraining(ViTMAEForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vit = WsiMAEModel(config)
        self.decoder = WsiMAEDecoder(config, num_patches=self.vit.embeddings.num_patches)
        self.post_init()
        # опциональная автозагрузка предобученных весов
        if getattr(self.config, "is_load_pretrained", False):
            print("Load weights....: ")
            if not self.config.pretrained_model_name:
                self.config.pretrained_model_name ="facebook/vit-mae-base"
            print("self.config.pretrained_model_name", self.config.pretrained_model_name)
            try:
                # self.vit =self.vit.from_pretrained(
                #     "facebook/vit-mae-base"  # полезно, если размеры отличаются
                # )
                print("Loading pretrained MAE weights from 'facebook/vit-mae-base'...")
                base = ViTMAEForPreTraining.from_pretrained(
                    self.config.pretrained_model_name, config = self.config  # полезно, если размеры отличаются
                )
                missing, unexpected = self.load_state_dict(base.state_dict(), strict=False)
                if missing:
                    print(f"Missing keys (first 10): {missing[:10]}{' ...' if len(missing)>10 else ''}")
                if unexpected:
                    print(f"Unexpected keys (first 10): {unexpected[:10]}{' ...' if len(unexpected)>10 else ''}")
                print("Pretrained weights loaded into WsiMAEForPreTraining.")
            except Exception as e:
               raise(f"Failed to load pretrained weights: {e}")
               

        # # важный хук HF (реинициализация голов и т.п., если нужно)
        # self.post_init()
        
    def patchify(self, imgs, interpolate_pos_encoding: bool = False):
        p = self.config.patch_size
        b, n, c, h, w = imgs.shape
        assert h == w == self.config.image_size 
        assert h % p == 0 and w % p == 0
        patches = rearrange(imgs, 'b n c h w -> (b n) c h w')
        patches = super().patchify(patches, interpolate_pos_encoding=interpolate_pos_encoding)
        return patches
    

    
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
    
        print("self.config.norm_pix_loss:", self.config.norm_pix_loss)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
  
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        
        return loss
    
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

        vit_out = self.vit(wsi_values)
        
        cls_tokens = vit_out.last_hidden_state[:, 0, :]  # [B_new, hidden_size]
        
        if self.use_transformer_pool:
            # Reshape classification tokens and get patient representation
            features = cls_tokens.unsqueeze(1)  # [B_new, 1, hidden_size]
            
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

            x = self.projection(cls_tokens)

            
        return x.squeeze(-1)