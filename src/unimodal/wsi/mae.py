import torch
import torch.nn as nn
from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEModel,
    ViTMAEDecoder,
    ViTMAEForPreTraining,
    ViTMAEModelOutput)
from typing import Callable, Optional, Union
from einops import rearrange, reduce

class WsiMAEModel(ViTMAEModel):
    def __init__(self, config):
        super().__init__(config)
        
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
                                  noise=None,
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
            try:
                print("Loading pretrained MAE weights from 'facebook/vit-mae-base'...")
                base = ViTMAEForPreTraining.from_pretrained(
                    "facebook/vit-mae-base"  # полезно, если размеры отличаются
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
    
    def unpatchify(self, patchified_pixel_values, original_image_size: Optional[tuple[int, int]] = None, batch_size: int =1):
        unpatchified = super().unpatchify(patchified_pixel_values, original_image_size)
        patches = rearrange(unpatchified, '(b n) c h w -> b n c h w', b=batch_size)
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