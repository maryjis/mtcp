import torch.nn as nn
import torch
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModel,ViTMAEDecoder, ViTMAEForPreTraining, get_1d_sincos_pos_embed_from_grid
import numpy as np
from typing import Optional, Set, Tuple, Union
import random
import json
from torch.nn.utils.rnn import pad_sequence
"""
This classes adopted from  https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit_mae/modeling_vit_mae.py#L323
"""

class RnaMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, cfg, columns_order=None):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        if cfg.patch_embedding["architecture"] == "tmae":
            self.patch_embeddings = RnaTMAEPatchEmbeddings(cfg)
        elif cfg.patch_embedding["architecture"] == "clusterd_go":
            self.patch_embeddings = RnaClusterdGOPatchEmbeddings(cfg, columns_order)
        else:
            raise ValueError(f"Invalid patch embedding type: {cfg.patch_embedding.type}")
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, cfg.hidden_size), requires_grad=False
        )
        self.patch_size = cfg.patch_size
        self.config = cfg
        self.initialize_weights()
        
    def init_deepsets_weights(self):
        # Инициализация всех Linear в phi
        for layer in self.patch_embeddings.deepsets.phi.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

        # Инициализация всех Linear в rho
        for layer in self.patch_embeddings.deepsets.rho.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
                
    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.position_embeddings.shape[-1], 
            np.arange(int(self.patch_embeddings.num_patches), dtype=np.float32)     )
        pos_embed = np.concatenate([np.zeros([1, self.position_embeddings.shape[-1]]), pos_embed], axis=0)
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # if self.config.patch_embedding["architecture"] == "tmae":
        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # elif self.config.patch_embedding["architecture"] == "clusterd_go":
        #     self.init_deepsets_weights()

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
        if isinstance(self.config.mask_ratio, list):
            mask_ratio = random.choice(self.config.mask_ratio)
            len_keep = int(seq_length * (1 - mask_ratio))
        else:
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



class RnaClusterdGOPatchEmbeddings(nn.Module):
    def __init__(self, cfg, columns_order):
        super().__init__()
        self.cfg = cfg
        self.clusters= json.load(open(cfg.patch_embedding["clusters_path"]))
        self.num_clusters = len(self.clusters)
        
        rna_size, patch_size = cfg.size, cfg.patch_size
        num_channels, hidden_size = cfg.num_channels, cfg.hidden_size

        num_patches = rna_size // patch_size
        
        self.rna_size = rna_size
        

        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = self.num_clusters
        print("columns_order", len(columns_order))
        self.cluster_indices = self.prepare_cluster_indices(columns_order, self.clusters)
        self.cluster_mask, self.max_cluster_lenth = self.make_cluster_mask(self.cluster_indices, len(columns_order))
        self.projection = nn.Linear(self.max_cluster_lenth, hidden_size)
        

    def gather_clusters_padded(self, rna_values, cluster_indices):
        """
        rna_values: (B, 1, N)
        cluster_indices: dict[name -> list[int]]
        
        Returns:
            padded_clusters: (B, T, L_max)
            cluster_lens: list of cluster lengths
        """
        B, _, N = rna_values.shape
        device = rna_values.device

        # Remove the singleton channel dimension
        rna_values = rna_values.squeeze(1)  # (B, N)
        
        clusters = list(cluster_indices.values())
        T = len(clusters)

        # Gather all cluster tensors
        cluster_tensors = []
        cluster_lens = []

        for idxs in clusters:
            idxs = torch.tensor(idxs, dtype=torch.long, device=device)
            # extract gene values for these indices
            vals = rna_values[:, idxs]        # (B, cluster_len)
            cluster_tensors.append(vals)
            cluster_lens.append(len(idxs))

        # pad to max cluster length
        L_max = max(cluster_lens)
        padded = torch.zeros((B, T, L_max), device=device)

        for t, vals in enumerate(cluster_tensors):
            length = vals.shape[1]
            padded[:, t, :length] = vals      # zero-padded beyond actual length

        return padded, cluster_lens    
        
    def make_cluster_mask(self,cluster_indices, n_genes, device=None):
        """
        cluster_indices : dict[str, torch.LongTensor]  # индексы генов в каждом кластере
        n_genes         : int = len(columns_order)
        return          : torch.BoolTensor (T, N)
        """
        T = len(cluster_indices)
        max_cluster_lenth = 0
        mask = torch.zeros(T, n_genes, dtype=torch.bool)
        for t, idxs in enumerate(cluster_indices.values()):
            mask[t, idxs] = True
            if len(idxs) >max_cluster_lenth:
                max_cluster_lenth = len(idxs)
        return mask, max_cluster_lenth   
     
    def prepare_cluster_indices(self, columns_order, clusters):
        """
        columns_order : list[str] — порядок генов в данных
        clusters      : dict[str, {"genes": set}] — словарь кластеров

        Возвращает: dict {cluster_name: torch.LongTensor(indices)}
        """
        # Словарь gene -> index в columns_order
        gene2idx = {g: i for i, g in enumerate(columns_order)}

        cluster_indices = {}
        for name, genes in clusters.items():
            idxs = [gene2idx[g] for g in genes if g in gene2idx]
            if idxs:  # если нашлись индексы
                cluster_indices[name] = torch.tensor(sorted(idxs), dtype=torch.long)
        return cluster_indices

    def forward(self, rna_values):
        
        batch_size, num_channels, rna_size = rna_values.shape 
        # rna_values = rna_values.permute(0, 2, 1).contiguous()
        padded_clustered_rna, cluster_lens = self.gather_clusters_padded(rna_values, self.cluster_indices)
        print("padded_clustered_rna: ", padded_clustered_rna.shape)
        embeddings =self.projection(padded_clustered_rna)
        return embeddings
        

class RnaMAEModel(ViTMAEModel):
    def __init__(self, config, columns_order=None):
        super().__init__(config)
        self.embeddings = RnaMAEEmbeddings(config, columns_order)
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
        if self.config.patch_embedding["architecture"] == "tmae":

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
        elif self.config.patch_embedding["architecture"] == "clusterd_go":
            return self.patchify_clustered(rna_values)

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
        if cfg.patch_embedding["architecture"] == "tmae":
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
        elif cfg.patch_embedding["architecture"] == "clusterd_go":
            return self.unpatchify_clustered(patchified_rna_values)
    
    def patchify_clustered(self, rna_values):
        """
        Args:
            rna_values: (B, 1, N)
        Returns:
            padded_clustered_rna: (B, T, L_max)
            cluster_lens: list[int]
        """
        pe = self.embeddings.patch_embeddings
        if not isinstance(pe, RnaClusterdGOPatchEmbeddings):
            raise ValueError("patchify_clustered() доступен только для clusterd_go архитектуры")

        padded_patchifiers, cluster_lens = pe.gather_clusters_padded(rna_values, pe.cluster_indices)
        return padded_patchifiers

    def unpatchify_clustered(self, cluster_values):
        """
        Восстанавливает исходный rna_values (B, 1, N) из кластеров.
        При пересечении генов — усредняет значения.
        """
        pe = self.embeddings.patch_embeddings
        cluster_indices = pe.cluster_indices
        B, T, Lmax = cluster_values.shape
        N = pe.rna_size
        device = cluster_values.device

        # создаём нулевые тензоры для сумм и счётчиков
        reconstructed = torch.zeros(B, 1, N, device=device)
        counts = torch.zeros(B, 1, N, device=device)

        # вставляем значения обратно по индексам генов
        for t, idxs in enumerate(cluster_indices.values()):
            idxs = idxs.to(device)
            L = min(len(idxs), Lmax)
            reconstructed[:, 0, idxs[:L]] += cluster_values[:, t, :L]
            counts[:, 0, idxs[:L]] += 1

        counts[counts == 0] = 1  # чтобы не делить на ноль
        reconstructed = reconstructed / counts
        return reconstructed
        

class RnaMAEDecoder(ViTMAEDecoder):
    def __init__(self, config,num_patches, rna_patching=None):
        super().__init__(config,num_patches)
        if not isinstance(rna_patching, RnaClusterdGOPatchEmbeddings):
            self.decoder_pred = nn.Linear(
                config.decoder_hidden_size, config.patch_size * config.num_channels, bias=True
            )  # encoder to decoder
        else:
            self.decoder_pred = nn.Linear(
                config.decoder_hidden_size, rna_patching.max_cluster_lenth, bias=True
            ) 
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
    def __init__(self, config, columns_order=None):
        super().__init__(config)
        self.config = config

        self.vit = RnaMAEModel(config, columns_order)
        self.decoder = RnaMAEDecoder(config,num_patches =self.vit.embeddings.patch_embeddings.num_patches, rna_patching=self.vit.embeddings.patch_embeddings)

        # Initialize weights and apply final processing
        self.post_init()
        
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
        # print("target.shape: ", target.shape)
        # print("pred.shape", pred.shape)

        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        # print("loss.shape", loss.shape)
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # print("loss.shape", loss.shape)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # print("loss:", loss)
        return loss

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
        if self.config.patch_embedding["architecture"] == "tmae":

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
        elif self.config.patch_embedding["architecture"] == "clusterd_go":
            return self.patchify_clustered(rna_values) 

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
        if self.config.patch_embedding["architecture"] == "tmae":
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
        elif self.config.patch_embedding["architecture"] == "clusterd_go":
            return self.unpatchify_clustered(patchified_rna_values)
    
    def patchify_clustered(self, rna_values):
        """
        Args:
            rna_values: (B, 1, N)
        Returns:
            padded_clustered_rna: (B, T, L_max)
            cluster_lens: list[int]
        """
        pe = self.vit.embeddings.patch_embeddings
        if not isinstance(pe, RnaClusterdGOPatchEmbeddings):
            raise ValueError("patchify_clustered() доступен только для clusterd_go архитектуры")

        padded_patchifiers, cluster_lens = pe.gather_clusters_padded(rna_values, pe.cluster_indices)
        return padded_patchifiers

    def unpatchify_clustered(self, cluster_values):
        """
        Восстанавливает исходный rna_values (B, 1, N) из кластеров.
        При пересечении генов — усредняет значения.
        """
        pe = self.vit.embeddings.patch_embeddings
        cluster_indices = pe.cluster_indices
        B, T, Lmax = cluster_values.shape
        N = pe.rna_size
        device = cluster_values.device

        # создаём нулевые тензоры для сумм и счётчиков
        reconstructed = torch.zeros(B, 1, N, device=device)
        counts = torch.zeros(B, 1, N, device=device)

        # вставляем значения обратно по индексам генов
        for t, idxs in enumerate(cluster_indices.values()):
            idxs = idxs.to(device)
            L = min(len(idxs), Lmax)
            reconstructed[:, 0, idxs[:L]] += cluster_values[:, t, :L]
            counts[:, 0, idxs[:L]] += 1

        counts[counts == 0] = 1  # чтобы не делить на ноль
        reconstructed = reconstructed / counts
        return reconstructed


class RnaSurvivalModel(nn.Module):
    def __init__(self, config, columns_order=None):
        super().__init__()
        self.config =config
        if config.is_load_pretrained:
            print("RnaSurvivalModel", self.config.pretrained_model_path)
            self.vit = RnaMAEModel.from_pretrained(self.config.pretrained_model_path, config = self.config)
        else:
            self.vit = RnaMAEModel(config, columns_order)
        self.projection = nn.Linear(self.config.hidden_size, self.config.output_dim)
        
    def forward(self, rna_values, masks=None):
        x = self.vit(rna_values)
        x = self.projection(x.last_hidden_state[:,0,:])
        return x.squeeze(-1)
    
    
    
def initialise_rna_mae_model(cfg, columns_order=None):
    return RnaSurvivalModel(cfg, columns_order)