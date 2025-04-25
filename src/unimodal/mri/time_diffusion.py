from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEForPreTrainingOutput,
    ViTMAEModelOutput
)
import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
from omegaconf import OmegaConf
from typing import Optional, Tuple, Union

from .mae import MriTMAEPatchEmbeddings
from .diffusion import MriDiffusionEmbeddings, MriDiffusionModel, MriDiffusionDecoder, MriDiffusionForPreTraining, MriMaeSurvivalModel

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device

        # We use half the dimension because we'll generate sin and cos for each dimension, which will then be concatenated
        half_dim = self.dim // 2

        # This creates a constant used to space out the frequency bands.
        # The constant 10000 comes from the original Transformer paper and works well in practice.
        # Dividing by `(half_dim - 1)` ensures that the frequencies span from 1 to 10000 evenly in log-space
        embeddings = np.log(10000) / (half_dim - 1)
        # print(embeddings)

        # This creates a tensor [0, 1, ..., half_dim-1].
        # Multiplying by -embeddings and then applying exp creates a tensor of decreasing values from 1 to 1/10000.
        # This generates the frequency bands for the sinusoidal functions
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # print(embeddings)

        # This multiplies each timestep by all frequency bands and adds a dimension to embeddings, making it a row vector.
        # The result is a 2D tensor where each row corresponds to a timestep, and each column to a frequency
        embeddings = time[:, None] * embeddings[None, :]
        # print(embeddings)

        # This applies sin and cos functions to the embeddings.
        # The results are concatenated along the last dimension.
        # This gives the final embedding where odd indices are sin and even indices are cos
        # NOTE FROM ME: i < dim/2 are sin and i >= dim/2 are cos
        # print("sin", embeddings.sin(), "cos", embeddings.cos())
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class MriTimeDiffusionPatchEmbeddings(MriTMAEPatchEmbeddings):
    """
    This class turns `mri_values` of shape `(batch_size, channels, mri_size, mri_size, mri_size)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        assert cfg.mri_size%cfg.patch_size==0, "Mri size must be divisible by patch size"

        self.mri_size = cfg.mri_size
        self.patch_size = cfg.patch_size
        self.num_channels = cfg.num_channels
        self.num_patches = cfg.mri_size**3 // cfg.patch_size**3

        if cfg.to_dict().get("embeddings_layers", None):
            c = OmegaConf.create(cfg.to_dict())
            self.projection = nn.ModuleList([getattr(nn, layer["name"])(*layer.get("args", []), **layer.get("kwargs", {})) for layer in c["embeddings_layers"]])
            self.convs_idxs = [i for i, layer in enumerate(c["embeddings_layers"]) if layer["name"] == "Conv3d"]
            self.time_convs_idxs = self.convs_idxs[:-1:2]
            self.time_projectors = nn.ModuleDict({str(i): nn.Linear(cfg.time_embeddging_dim, c["embeddings_layers"][i]["args"][1]) for i in self.convs_idxs[:-1:2]})
            self.time_act = nn.ReLU()
        else:
            print("WARNING: you haven't parametrized encoder embeddings layers in model config, heavy default convolution is used")
            self.projection = nn.Conv3d(cfg.num_channels, cfg.hidden_size, kernel_size=cfg.patch_size, stride=cfg.patch_size)

    def forward(self, mri_values, time_embeddings):
        batch_size, num_channels, mri_size1, mri_size2, mri_size3 = mri_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                f"Make sure that the channel dimension of the mri values {num_channels} match with the one set in the configuration {self.num_channels}."
            )
          
        x = rearrange(
            mri_values, 
            'b c (x p_x) (y p_y) (z p_z) -> (b x y z) c p_x p_y p_z', 
            p_x=self.patch_size, 
            p_y=self.patch_size, 
            p_z=self.patch_size
        )
        # x = self.projection(x) # B*S,C,P_x,P_y,P_z -> B*S,E,1,1,1
        if isinstance(self.projection, nn.ModuleList):
            for i, layer in enumerate(self.projection):
                x = layer(x)
                if str(i) in self.time_projectors:
                    time_emb = self.time_act(self.time_projectors[str(i)](time_embeddings))
                    time_emb = time_emb[(..., ) + (None, ) * len(x.shape[2:])]
                    time_emb = time_emb.repeat(
                        x.shape[0]//time_emb.shape[0], 
                        *([1]*len(time_emb.shape[1:]))
                    )
                    x = x + time_emb
        else:
            x = self.projection(x)

        x = rearrange(x.squeeze(), '(b s) e -> b s e', b=batch_size)
        return x

class MriTimeDiffusionEmbeddings(MriDiffusionEmbeddings):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.patch_embeddings = MriTimeDiffusionPatchEmbeddings(cfg)
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, cfg.hidden_size), requires_grad=False
        )
        self.patch_size = cfg.patch_size
        self.config = cfg
        self.initialize_weights()

    def forward(self, mri_values, time_embeddings, noise=None, interpolate_pos_encoding: bool = False):
        batch_size, num_channels, mri_size1, mri_size2, mri_size3 = mri_values.shape
        embeddings = self.patch_embeddings(mri_values, time_embeddings)
        
        # add position embeddings w/o cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]
     
        embeddings, mask, ids_restore = self.random_nothing(embeddings)

        # append cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore
    
class MriTimeDiffusionModel(MriDiffusionModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = MriTimeDiffusionEmbeddings(config)
        # Time embedding
        self.time_embeddging_dim = config.time_embeddging_dim
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(config.time_embeddging_dim),
                nn.Linear(config.time_embeddging_dim, config.time_embeddging_dim),
                nn.ReLU()
            )
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        times: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple, ViTMAEModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ViTMAEModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        >>> model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        time_embeddings = self.time_mlp(times)

        embedding_output, mask, ids_restore = self.embeddings(
            pixel_values, time_embeddings, noise=noise, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]

        return ViTMAEModelOutput(
            last_hidden_state=sequence_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class MriTimeDiffusionForPreTraining(MriDiffusionForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.vit = MriTimeDiffusionModel(config)
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
            times=alpha_t,
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