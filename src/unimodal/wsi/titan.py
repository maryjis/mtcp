import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModelOutput

logger = logging.getLogger(__name__)

TITAN_DEFAULT_FEATURE_DIM = 768
TITAN_DEFAULT_PATCH_SIZE_LV0 = 512


class _TitanEmbeddingProxy:
    """Mimics embeddings.num_patches for MultiMAEModel compatibility."""
    def __init__(self, num_patches: int):
        self.num_patches = num_patches


# ---------------------------------------------------------------------------
#  Pre-computed slide-level embeddings (frozen wrapper)
# ---------------------------------------------------------------------------

class TitanEmbeddingEncoder(nn.Module):
    """
    Wraps pre-computed TITAN slide-level embeddings so they can be used
    as a frozen encoder inside the multimodal pipeline.

    Input:  [B, 1, embed_dim]   (pre-computed TITAN embedding per sample)
    Output: ViTMAEModelOutput  (cls + 1 token → seq_len = 2)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.num_tokens = 1

        self.embeddings = _TitanEmbeddingProxy(num_patches=self.num_tokens)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.layernorm = nn.LayerNorm(embed_dim)

        nn.init.normal_(self.cls_token, std=config.initializer_range)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        noise: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> ViTMAEModelOutput:
        if pixel_values.dim() == 2:
            pixel_values = pixel_values.unsqueeze(1)

        B = pixel_values.shape[0]

        x = self.projection(pixel_values)
        x = self.layernorm(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        seq_len = self.num_tokens
        mask = torch.zeros(B, seq_len, device=x.device)
        ids_restore = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(B, -1)

        return ViTMAEModelOutput(
            last_hidden_state=x,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=None,
            attentions=None,
        )

    def patchify(self, x, interpolate_pos_encoding: bool = False):
        return x

    def unpatchify(self, x, original_size=None):
        return x


class TitanEmbeddingSurvival(nn.Module):
    """
    Perceptron head on top of frozen TITAN embeddings for survival prediction.
    Only the head is trained; TITAN embeddings are loaded from disk.

    Input:  [B, 1, embed_dim]
    Output: [B, output_dim]   (hazard logits)
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, final_dropout: float,
                 output_dim: int = 20):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(final_dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, wsi_embeddings, masks=None):
        if wsi_embeddings.dim() == 3:
            wsi_embeddings = wsi_embeddings.squeeze(1)
        return self.head(wsi_embeddings)


# ---------------------------------------------------------------------------
#  Live TITAN model — loads actual weights, processes patch features on GPU
# ---------------------------------------------------------------------------

def _load_titan_model(model_path: str):
    """Load the TITAN HuggingFace model (vision + text encoders)."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    logger.info("TITAN model loaded from %s", model_path)
    return model


class TitanLiveEncoder(nn.Module):
    """
    Live TITAN encoder: loads actual model weights and processes
    patch-level CONCHv1.5 features + coordinates into a slide embedding.

    The input tensor is a *packed* representation:
        pixel_values: [B, max_patches, feature_dim + 2]
          - [:, :, :feature_dim]        — CONCHv1.5 patch features  (768-d)
          - [:, :, feature_dim:feature_dim+2] — patch coordinates (x, y)
        Zero-padded rows are automatically filtered out.

    Output: ViTMAEModelOutput  (cls + 1 slide-token → seq_len = 2)
    """

    def __init__(self, config, titan_model_path: str = "MahmoodLab/TITAN",
                 patch_size_lv0: int = TITAN_DEFAULT_PATCH_SIZE_LV0,
                 feature_dim: int = TITAN_DEFAULT_FEATURE_DIM):
        super().__init__()
        self.config = config
        self.patch_size_lv0 = patch_size_lv0
        self.feature_dim = feature_dim
        self.num_tokens = 1
        self.embeddings = _TitanEmbeddingProxy(num_patches=self.num_tokens)

        self.titan = _load_titan_model(titan_model_path)
        for p in self.titan.parameters():
            p.requires_grad = False

        embed_dim = config.hidden_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=config.initializer_range)

    # ----- internal helpers -----

    @torch.no_grad()
    def _encode_single_slide(self, features: torch.Tensor,
                             coords: torch.Tensor) -> torch.Tensor:
        """Run TITAN vision encoder on a single slide. Returns [feature_dim]."""
        with torch.autocast(device_type=features.device.type,
                            dtype=torch.bfloat16,
                            enabled=features.device.type == "cuda"):
            emb = self.titan.encode_slide_from_patch_features(
                features, coords, self.patch_size_lv0,
            )
        if emb.dim() > 1:
            emb = emb.squeeze(0)
        return emb.float()

    # ----- nn.Module API -----

    def forward(self, pixel_values: torch.FloatTensor,
                noise: Optional[torch.FloatTensor] = None,
                **kwargs) -> ViTMAEModelOutput:
        if pixel_values.dim() == 2:
            pixel_values = pixel_values.unsqueeze(0)

        B = pixel_values.shape[0]
        device = pixel_values.device

        feats = pixel_values[:, :, :self.feature_dim]
        coords = pixel_values[:, :, self.feature_dim:self.feature_dim + 2].long()

        slide_embs = []
        for i in range(B):
            feat_i = feats[i]
            valid = feat_i.abs().sum(dim=-1) > 0
            feat_valid = feat_i[valid].unsqueeze(0)
            coord_valid = coords[i][valid].unsqueeze(0)

            if feat_valid.shape[1] == 0:
                emb = torch.zeros(self.feature_dim, device=device)
            else:
                emb = self._encode_single_slide(feat_valid, coord_valid)
            slide_embs.append(emb)

        x = torch.stack(slide_embs, dim=0).unsqueeze(1)      # [B, 1, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)                 # [B, 2, D]

        seq_len = self.num_tokens
        mask = torch.zeros(B, seq_len, device=device)
        ids_restore = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)

        return ViTMAEModelOutput(
            last_hidden_state=x, mask=mask, ids_restore=ids_restore,
            hidden_states=None, attentions=None,
        )

    def patchify(self, x, interpolate_pos_encoding: bool = False):
        return x

    def unpatchify(self, x, original_size=None):
        return x


class TitanLiveSurvival(nn.Module):
    """
    Live TITAN encoder + MLP survival head.
    TITAN weights are frozen; only the MLP head is trained.

    Input:  packed tensor [B, max_patches, feature_dim + 2]
    Output: [B, output_dim]  (hazard logits)
    """

    def __init__(self, titan_model_path: str, embedding_dim: int,
                 hidden_dim: int, final_dropout: float,
                 output_dim: int = 20,
                 patch_size_lv0: int = TITAN_DEFAULT_PATCH_SIZE_LV0):
        super().__init__()
        self.feature_dim = embedding_dim
        self.patch_size_lv0 = patch_size_lv0

        self.titan = _load_titan_model(titan_model_path)
        for p in self.titan.parameters():
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(final_dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    @torch.no_grad()
    def _encode_slides(self, wsi_data: torch.Tensor) -> torch.Tensor:
        B = wsi_data.shape[0]
        device = wsi_data.device
        feats = wsi_data[:, :, :self.feature_dim]
        coords = wsi_data[:, :, self.feature_dim:self.feature_dim + 2].long()

        embs = []
        for i in range(B):
            feat_i = feats[i]
            valid = feat_i.abs().sum(dim=-1) > 0
            feat_valid = feat_i[valid].unsqueeze(0)
            coord_valid = coords[i][valid].unsqueeze(0)

            if feat_valid.shape[1] == 0:
                emb = torch.zeros(self.feature_dim, device=device)
            else:
                with torch.autocast(device_type=device.type,
                                    dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    emb = self.titan.encode_slide_from_patch_features(
                        feat_valid, coord_valid, self.patch_size_lv0,
                    )
                if emb.dim() > 1:
                    emb = emb.squeeze(0)
            embs.append(emb.float())
        return torch.stack(embs, dim=0)

    def forward(self, wsi_data: torch.Tensor, masks=None) -> torch.Tensor:
        if wsi_data.dim() == 2:
            wsi_data = wsi_data.unsqueeze(0)
        slide_embeddings = self._encode_slides(wsi_data)
        return self.head(slide_embeddings)
