import json
import random 
import os
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModelOutput, ViTMAEForPreTrainingOutput
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from transformers import PreTrainedModel
from flamingo_pytorch import PerceiverResampler

from omegaconf import DictConfig, OmegaConf

from src.unimodal.rna.mae import RnaMAEModel, RnaMAEDecoder
from src.unimodal.mri.mae import MriMAEModel
from src.unimodal.dna.models import DNAmSurvivalModel, DNAmMAEModel
from src.unimodal.wsi.mae import WsiMAEModel
from src.unimodal.cnv.models import CNVMAEModel
from src.unimodal.clinical.models import ClinicalModel
from src.unimodal.mri.mae import MriMAEDecoderPred
from src.multimodal.losses import CLIPAlignmentLoss
from src.utils import ExperimentTracker


@dataclass
class MultiModalOutput(ViTMAEModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    mask: Optional[torch.LongTensor] = None
    ids_restore: Optional[torch.LongTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None

    last_hidden_state_with_cls: Optional[torch.FloatTensor] = None
    modality_token_intervals: Optional[Dict[str, Tuple[int, int]]] = None

class BatchSigmaClipper(nn.Module):
    def __init__(self, k: float = 5.0, eps: float = 1e-8, nan_safe: bool = True):
        super().__init__()
        self.k = float(k)
        self.eps = eps
        self.nan_safe = nan_safe

    def forward(self, x: torch.Tensor):
        # x: [B, T, D]
        if self.nan_safe:
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        # одно среднее и одно std по всему батчу и всем признакам
        mean = x.mean(dim=(0,1,2))  # [1,1,1]
        std  = x.std(dim=(0,1,2), unbiased=False).clamp_min(self.eps)  # [1,1,1]

        lower = mean - self.k * std
        upper = mean + self.k * std

    
        x = torch.clamp(x, min=lower, max=upper)

        return x
    
class UnimodalEncoder(nn.Module):
    def __init__(self, encoder, unimodal_hidden_size,
                 multimodal_hidden_size = None, is_projection = False,
                 modality =None, tracker: ExperimentTracker =None):

        super().__init__()
        self.encoder = encoder
        self.inner_size = unimodal_hidden_size
        self.is_projection = is_projection
        self.modality =modality
        self.tracker =tracker
        
        if self.is_projection:
            if multimodal_hidden_size is None:
                raise ValueError("multimodal_hidden_size must be provided when is_projection=True")
            self.projection = nn.Linear(unimodal_hidden_size, multimodal_hidden_size)
            
    def forward(self, x):
        # Remove debug print statement

        x = self.encoder(x)
        
        if self.is_projection:
            # Create new object instead of modifying in place
            x = ViTMAEModelOutput(
                last_hidden_state=self.projection(x.last_hidden_state),
                hidden_states=x.hidden_states,
                attentions=x.attentions,
                mask=x.mask
            )
        return x

class MultiMAEDecoder(RnaMAEDecoder):
    def __init__(self, config, num_patches):
        super().__init__(config, num_patches)
        self.modalities = config.modalities
        if not config.postprocessing:
            self.decoder_pred = nn.Linear(
                config.decoder_hidden_size, config.patch_size * config.num_channels, bias=True
            )  # encoder to decoder
        else:
             self.decoder_pred = nn.Identity()
                 
        self.initialize_weights(num_patches)
        

class AttentionPooling(nn.Module):
    """Пулинг токенов внутри модальности с помощью attention."""
    def __init__(self, embed_dim):
        super().__init__()
        self.att_vector = nn.Parameter(torch.randn(embed_dim))

    def forward(self, x):
        # x: [B, T, D]
        # вычисляем веса внимания для каждого токена
        attn_scores = torch.matmul(x, self.att_vector)  # [B, T]
        attn_weights = F.softmax(attn_scores, dim=1)    # [B, T]
        # агрегируем токены
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [B, D]
        return pooled.unsqueeze(1)
    
class TopKPooling(nn.Module):
    def __init__(self, embed_dim, k=20):
        super().__init__()
        self.k = k
        self.scorer = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: [B, T, D]
        scores = self.scorer(x).squeeze(-1)       # [B, T]
        topk = torch.topk(scores, self.k, dim=1).indices
        B = x.size(0)
        out = torch.stack([x[b, idx] for b, idx in enumerate(topk)], dim=0)
        return out  # [B, k, D] 
    
    
class MultiMAEModel(PreTrainedModel):
     
    def __init__(self, cfg, tracker=None, preproc=None):
        super().__init__(cfg)
        self.cfg = cfg
        self.tracker = tracker

        self.modalities = cfg.modalities
        self.__init_encoders__(preproc)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.encoders = nn.ModuleDict(self.encoders)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.encoder_fusion_strategy = None
        if self.cfg.make_modality_pooling:
            self.polings = nn.ModuleDict({modality: TopKPooling(cfg.hidden_size, k=20) for modality in  cfg.modalities})
        if self.cfg.make_modality_normalizers:
            self.normalizers = nn.ModuleDict({modality: nn.LayerNorm(cfg.hidden_size) for modality in  cfg.modalities})
        if self.cfg.make_modality_clipper:
            self.cliper = BatchSigmaClipper(k=4.0)
        

        if self.cfg.encoder_fusion_strategy =="masked_attention":
            self.encoder_fusion_strategy = MaskAttentionFusion(cfg.encoder_fusion_depth, cfg.encoder_fusion_dim,
                                                                cfg.encoder_fusion_nhead,
                                                        cfg.encoder_fusion_dim_feedforward, cfg.encoder_fusion_dropout, cfg.encoder_activation_name)
        self.initialize_weights()

    def __init_encoders__(self, preproc=None):
        """Initialize encoders for each modality."""
        self.encoders = {}
        for modality in self.modalities:
            if modality == "rna":
                cfg_rna_model = ViTMAEConfig(**self.cfg.rna_model)
                encoder = None
                column_order = preproc["rna"].get_column_order() if preproc else None
                if cfg_rna_model.is_load_pretrained:
                    encoder = RnaMAEModel.from_pretrained(cfg_rna_model.pretrained_model_path, config=cfg_rna_model, column_order=column_order)
                    for param in encoder.parameters():
                        param.requires_grad = False
                else:
                    encoder = RnaMAEModel(cfg_rna_model, column_order)
                
                self.encoders[modality] = UnimodalEncoder(
                    encoder,
                    cfg_rna_model.hidden_size, 
                    self.cfg.hidden_size,
                    self.cfg.is_projection,
                    modality
                )
            elif modality == "mri":
                cfg_mri_model = ViTMAEConfig(**self.cfg.mri_model)
                encoder = None
                if cfg_mri_model.is_load_pretrained:
                    encoder = MriMAEModel.from_pretrained(cfg_mri_model.pretrained_model_path, config=cfg_mri_model)
                    for param in encoder.parameters():
                        param.requires_grad = False
                else:
                    encoder = MriMAEModel(cfg_mri_model)
                
                self.encoders[modality] = UnimodalEncoder(
                    encoder, 
                    cfg_mri_model.hidden_size, 
                    self.cfg.hidden_size,
                    self.cfg.is_projection,
                    modality
                )
            elif modality == "dnam":
                cfg_dnam_model = ViTMAEConfig(**self.cfg.dnam_model)
                encoder = None
                column_order = preproc["dnam"].get_column_order() if preproc else None
                if cfg_dnam_model.is_load_pretrained:
                    encoder = DNAmMAEModel.from_pretrained(cfg_dnam_model.pretrained_model_path, config=cfg_dnam_model, column_order=column_order)
                    for param in encoder.parameters():
                        param.requires_grad = False
                else:
                    encoder = DNAmMAEModel(cfg_dnam_model, column_order)
                    
                self.encoders[modality] = UnimodalEncoder(
                    encoder,
                    cfg_dnam_model.hidden_size, 
                    self.cfg.hidden_size,
                    self.cfg.is_projection,
                    modality
                )
            elif modality == "cnv":
                cfg_cnv_model = ViTMAEConfig(**self.cfg.cnv_model)
                encoder = None
                if cfg_cnv_model.is_load_pretrained:
                    encoder = CNVMAEModel.from_pretrained(cfg_cnv_model.pretrained_model_path, config=cfg_cnv_model)
                    for param in encoder.parameters():
                        param.requires_grad = False
                else:
                    encoder = CNVMAEModel(cfg_cnv_model)
                    
                self.encoders[modality] = UnimodalEncoder(
                    encoder,
                    cfg_cnv_model.hidden_size, 
                    self.cfg.hidden_size,
                    self.cfg.is_projection,
                    modality
                )
            elif modality == "wsi":
                print("self.cfg.wsi_model: ", self.cfg.wsi_model)
                cfg_wsi_model = ViTMAEConfig(**self.cfg.wsi_model)
                encoder = None
                
                print("cfg_wsi_model.is_load_pretrained: ", cfg_wsi_model)
                if cfg_wsi_model.is_load_pretrained:
                    print("cfg_wsi_model.pretrained_model_path: ", cfg_wsi_model.pretrained_model_path)
                    # encoder = WSIEmbeddingMAEModel.from_pretrained(cfg_wsi_model.pretrained_model_path, config=cfg_wsi_model)
                    if cfg_wsi_model.pretrained_model_name == "":
                        cfg_wsi_model.pretrained_model_path = "facebook/vit-mae-base"

                        
                    encoder = WsiMAEModel.from_pretrained(cfg_wsi_model.pretrained_model_path, config=cfg_wsi_model)
                    
                    for param in encoder.parameters():
                        print("Freezing: ")
                        param.requires_grad = False
                    # else:
                    #     print(os.getcwd())
                    #     print(cfg_wsi_model.pretrained_model_path)
                    #     state_dict = torch.load(cfg_wsi_model.pretrained_model_path)
                    #     print("State dict:", state_dict)
                    #     encoder = WsiMAEModel(config=cfg_wsi_model)
                    #     encoder.load_state_dict(state_dict)
                    # # for param in encoder.parameters():

                    #     param.requires_grad = False
                else:
                    #encoder = WSIEmbeddingMAEModel(config =cfg_wsi_model)
                    encoder = WsiMAEModel(config =cfg_wsi_model)
                    
                self.encoders[modality] = UnimodalEncoder(
                    encoder,
                    cfg_wsi_model.hidden_size, 
                    self.cfg.hidden_size,
                    self.cfg.is_projection,
                    modality
                )
            elif modality == "clinical":
                cfg_clinical_model = ViTMAEConfig(**self.cfg.clinical_model)
                if self.cfg.clinica_linear_last_fusion:
                    pass
                else:
                    encoder = ClinicalModel(config =cfg_clinical_model)
                    self.encoders[modality] = UnimodalEncoder(
                                        encoder,
                                        cfg_clinical_model.hidden_size, 
                                        self.cfg.hidden_size,
                                        self.cfg.is_projection,
                                        modality)  
            else:
                # Add support for other modalities
                raise NotImplementedError(f"Encoder for modality {modality} not implemented")
            
    def get_patches_number(self, modality: str) -> int:
        """Get number of patches for a specific modality.
        
        Args:
            modality: The modality type (e.g. "rna")
            
        Returns:
            Number of patches for the modality
        """
        if modality =="clinical":
            return self.encoders[modality].encoder.total_tokens
        else:
            return self.encoders[modality].encoder.embeddings.num_patches

    def get_cls_ids(self):
        cls_indexes =[1]
        for modality in self.modalities[1:]:
            if modality =="clinical":
                if self.cfg.clinica_linear_last_fusion:
                    continue
                else:
                    num_tokens= self.encoders[modality].encoder.total_tokens
            else:
                num_tokens= self.encoders[modality].encoder.embeddings.num_patches
            cls_indexes.append(num_tokens+1)
        return cls_indexes
     
    def get_all_patches_number(self) -> int:
        """Calculate total number of patches across all modalities.
        
        Returns:
            Total number of patches
        """
        num_patches = sum(self.get_patches_number(modality) for modality in self.modalities)
        return num_patches
            
    def batch_embeddings(self, sample: torch.FloatTensor, mask: torch.BoolTensor, modality: str, multimodal_lenths: int) -> ViTMAEModelOutput:
        """Process embeddings for a batch of samples for a given modality.
        
        Args:
            sample: Input tensor of shape (batch_size, num_channels, sample_size)
            mask: Boolean mask tensor indicating which samples to mask
            modality: Name of the modality being processed
            multimodal_lenths: Cumulative sequence length of previous modalities
            
        Returns:
            ViTMAEModelOutput containing the processed embeddings
        """
        batch_size = sample.shape[0]
        seq_length = self.get_patches_number(modality)
        device = sample.device

   
        # If mask is all True, process entire batch normally
        if torch.all(mask):
            
            embedded_sample = self.encoders[modality](sample)
            # Offset ids_restore by cumulative sequence length

            embedded_sample.ids_restore = embedded_sample.ids_restore + multimodal_lenths

            return embedded_sample

        elif torch.any(mask):

            # Get indices of masked and unmasked samples
            empty_sample_ids = torch.nonzero(~mask, as_tuple=True)[0].to(device)
          # Create tensors for empty samples
            mask_empty = torch.ones(len(empty_sample_ids), seq_length, device=device)
            ids_restore_empty = torch.arange(seq_length, device=device).repeat(len(empty_sample_ids), 1) + multimodal_lenths  
                      
            non_empty_sample_ids = torch.nonzero(mask, as_tuple=True)[0].to(device)
           
            # Process non-empty samples
            non_empty_samples = torch.index_select(sample, dim=0, index=non_empty_sample_ids)
            
            embedded_non_empty = self.encoders[modality](non_empty_samples)
            
            
            empty_sequence = self.mask_token.repeat(len(empty_sample_ids), embedded_non_empty.last_hidden_state.shape[1], 1)
           
            # Combine empty and non-empty tensors
            last_hidden_state = torch.cat([
                embedded_non_empty.last_hidden_state,
                empty_sequence
            ], dim=0)
            
            mask_combined = torch.cat([
                embedded_non_empty.mask,
                mask_empty
            ], dim=0)
           
            

            ids_restore = torch.cat([
                embedded_non_empty.ids_restore + multimodal_lenths,
                ids_restore_empty
            ], dim=0)
            

            # Reorder tensors based on original sample ordering
            #indices = torch.cat([empty_sample_ids, non_empty_sample_ids]).sort()[1]
            indices = torch.argsort(torch.cat([non_empty_sample_ids, empty_sample_ids]), dim=0)
            
          
            last_hidden_state = torch.index_select(last_hidden_state, dim=0, index=indices)
            mask_combined = torch.index_select(mask_combined, dim=0, index=indices)
            ids_restore = torch.index_select(ids_restore, dim=0, index=indices)
            
           
            return ViTMAEModelOutput(
                last_hidden_state=last_hidden_state,
                mask=mask_combined,
                ids_restore=ids_restore,
                hidden_states=None,
                attentions=None
            )
        else:
            empty_sample_ids = torch.nonzero(~mask, as_tuple=True)[0].to(device)
            # Create tensors for empty samples
            mask_empty = torch.ones(len(empty_sample_ids), seq_length, device=device)
            ids_restore_empty = torch.arange(seq_length, device=device).repeat(len(empty_sample_ids), 1) + multimodal_lenths 
            if isinstance(self.encoders[modality].encoder.config.mask_ratio, list):
                mask_ratio = random.choice(self.encoders[modality].encoder.config.mask_ratio)
                len_keep = int(seq_length * (1 - mask_ratio))
            else:
                len_keep = int(seq_length * (1 - self.encoders[modality].encoder.config.mask_ratio))
             
            empty_sequence = self.mask_token.repeat(len(empty_sample_ids),len_keep+1, 1)
            return ViTMAEModelOutput(
                last_hidden_state=empty_sequence,
                mask=mask_empty,
                ids_restore=ids_restore_empty,
                hidden_states=None,
                attentions=None
            )

    def save_token_visualisation(self, modality: str, last_hidden_state: torch.Tensor, step: int = None):
        """
        modality: str - название модальности (например "text")
        last_hidden_state: (batch_size, n_tokens, hidden_size) - эмбеддинги для этой модальности
        step: int - шаг обучения (если нужен лог как серия)
        """
        # усредняем по batch_size и n_tokens → (hidden_size,)
        emb = last_hidden_state.mean(dim=(0, 1)).cpu().detach().numpy()
        # строим гистограмму
        plt.figure()
        plt.hist(emb, bins=30, alpha=0.7, color="steelblue")
        plt.title(f"Distribution for {modality}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        # логируем напрямую в Neptune
        self.tracker.log_image(
            key=f"visualisations/{modality}",
            image=plt.gcf()
        )

        plt.close()
        
               

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=self.cfg.initializer_range)
        # torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)
    
    def dropout(self, last_hidden_state, modality):
        """
        last_hidden_state: (B, T, D)
        Выводит индексы семплов, которые были дропнуты для данной модальности.
        """
        if self._md_active is None:
            return last_hidden_state

        p_mod = self.cfg.modality_dropout_p.get(modality, 0.0)
        if p_mod == 0.0:
            return last_hidden_state

        B = last_hidden_state.shape[0]

        # сэмплируем для каждого примера
        rand = torch.rand(B, device=last_hidden_state.device)

        # условия:
        # 1) пример активен
        # 2) dropout ещё не применялся
        # 3) прошли вероятность модальности
        mask = (
            self._md_active
            & (~self._md_used)
            & (rand < p_mod)
        )

        dropped_indices = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
        n_dropped = len(dropped_indices)

        if n_dropped > 0:
            self._md_used = self._md_used | mask
            
            # зануляем selected samples
            last_hidden_state = last_hidden_state * (~mask).view(B, 1, 1)
            
            # логируем
            print(f"[ModDrop] Dropped {n_dropped} samples for modality '{modality}': indices {dropped_indices}")

        return last_hidden_state

        
    def forward(
        self,
        x: Dict[str, torch.FloatTensor],
        masks: Dict[str, torch.FloatTensor],
        interpolate_pos_encoding: bool = False,
        return_attention: bool = False,
    ):
        """Forward pass through the model.
        
        Processes each modality separately, concatenates their embeddings,
        and passes through the decoder to get reconstructions.
        """
        # Track cumulative sequence length across modalities
        multimodal_length = 0
        encoder_outputs = []
        modality_token_lengths: Dict[str, int] = {}

        # Process each modality
        is_first = True
        # print("self.cfg.make_modality_dropout: ", self.cfg.make_modality_dropout)
        if self.training and self.cfg.make_modality_dropout:
            B = next(iter(x.values())).shape[0]

            # какие примеры вообще участвуют в modality dropout
            self._md_active = torch.rand(B, device=next(iter(x.values())).device) < self.cfg.p_dropout_value

            # для каких примеров уже применили dropout
            self._md_used = torch.zeros(B, dtype=torch.bool, device=self._md_active.device)
        else:
            self._md_active = None
            
        for modality in self.modalities:
            # print(modality)
            if modality =="clinical":
                if self.cfg.clinica_linear_last_fusion:
                    continue
            seq_length = self.get_patches_number(modality)
            sample = x[modality]
            

            mask = masks[modality]
            
            # Get embeddings for this modality
            embedded_sample = self.batch_embeddings(
                sample=sample,
                mask=mask,
                modality=modality,
                multimodal_lenths=multimodal_length
                
            )

            multimodal_length += seq_length
            
            #last_hidden_state = embedded_sample.last_hidden_state[:,1:, :]
            #try to take with token
            last_hidden_state = embedded_sample.last_hidden_state
 
            if modality =="wsi" and self.encoders["wsi"].encoder.config.max_patches_per_sample>1 and not self.encoders["wsi"].encoder.config.random_patch_selection:
                    n10, n_tokens, hidden_size = last_hidden_state.shape
                    n = n10 // self.encoders["wsi"].encoder.config.max_patches_per_sample

                    last_hidden_state = last_hidden_state.view(n, self.encoders["wsi"].encoder.config.max_patches_per_sample, n_tokens, hidden_size)  # (n, 10, n_tokens, hidden_size)
                    last_hidden_state = last_hidden_state.mean(dim=1)
                    embedded_sample.mask =embedded_sample.mask[:n,:]
                    embedded_sample.ids_restore =embedded_sample.ids_restore[:n,:]
                    
            if self.cfg.make_modality_dropout:
                    print("Make dropout:")
                    last_hidden_state =self.dropout(last_hidden_state, modality)
                            
            if self.cfg.make_modality_clipper:
                last_hidden_state =self.cliper(last_hidden_state)
                
            if self.cfg.make_simple_normalizer:
                last_hidden_state = F.normalize(last_hidden_state, p=2, dim=-1, eps=1e-8)
                    
            # self.save_token_visualisation(modality=modality,
            #             last_hidden_state=last_hidden_state)

            # print("last_hidden_state.shape: ", last_hidden_state.shape)
            # if self.cfg.per_modality_pooling[modality]:
            #     print("cfg.per_modality_pooling[modality]: ", self.cfg.per_modality_pooling[modality])
            #     last_hidden_state =self.polings[modality](embedded_sample.last_hidden_state[:,1:, :])
                
             
            # print("After normalize last_hidden_state.mean", torch.mean(last_hidden_state, dim=[0,1,2]))
            # last_hidden_state =self.normalizers[modality](last_hidden_state)
            # print("Before normalize last_hidden_state.mean", torch.mean(last_hidden_state, dim=[0,1,2]))  
            embedded_sample = ViTMAEModelOutput(
                    last_hidden_state=last_hidden_state,

                    mask=embedded_sample.mask,
                    ids_restore=embedded_sample.ids_restore,
                    hidden_states=embedded_sample.hidden_states,
                    attentions=embedded_sample.attentions
                )
            modality_token_lengths[modality] = int(last_hidden_state.shape[1])
            encoder_outputs.append(embedded_sample)
      

        # Combine embeddings from all modalities
        last_hidden_states = torch.cat([out.last_hidden_state for out in encoder_outputs], dim=1)
        masks = torch.cat([out.mask for out in encoder_outputs], dim=1)
        ids_restores = torch.cat([out.ids_restore for out in encoder_outputs],dim=1)
        

        cls_tokens = self.cls_token.expand(last_hidden_states.shape[0], -1, -1)
        
        class_token_mask = torch.full((masks.shape[0], 1), 0).to(masks.device)
        masks = torch.cat((class_token_mask, masks), dim=1) 
        
        last_hidden_states = torch.cat((cls_tokens, last_hidden_states), dim=1)
        

        modality_token_intervals: Dict[str, Tuple[int, int]] = {}
        token_start = 1  # 0 is global multimodal CLS token
        for modality in self.modalities:
            if modality == "clinical" and self.cfg.clinica_linear_last_fusion:
                continue
            token_len = modality_token_lengths[modality]
            token_end = token_start + token_len
            modality_token_intervals[modality] = (token_start, token_end)
            token_start = token_end

        fusion_attentions: Optional[Tuple[torch.Tensor, ...]] = None
        if self.encoder_fusion_strategy is not None:
            if last_hidden_states.shape[1] == masks.shape[1]:
                if return_attention:
                    last_hidden_states, fusion_attentions = self.encoder_fusion_strategy(
                        last_hidden_states,
                        masks,
                        return_attention=True,
                    )
                else:
                    last_hidden_states = self.encoder_fusion_strategy(last_hidden_states, masks)
            else:
                if return_attention:
                    last_hidden_states, fusion_attentions = self.encoder_fusion_strategy(
                        last_hidden_states,
                        None,
                        return_attention=True,
                    )
                else:
                    last_hidden_states = self.encoder_fusion_strategy(last_hidden_states, None)

        last_hidden_states_without_cls = last_hidden_states.clone()
        remove_cls = torch.ones(last_hidden_states_without_cls.size(1), dtype=torch.bool)
        indexes_for_remove = self.get_cls_ids()
        remove_cls[indexes_for_remove] = False
     
        last_hidden_states_without_cls_new = last_hidden_states_without_cls[:,remove_cls, :]

        concat_embedding = MultiModalOutput(
            last_hidden_state=last_hidden_states_without_cls_new,
            last_hidden_state_with_cls = last_hidden_states,
            mask=masks,
            ids_restore=ids_restores,
            hidden_states=None,
            attentions=fusion_attentions,
            modality_token_intervals=modality_token_intervals,
        )    
            
        return concat_embedding
        
        
class MultiMaeForPretraining(nn.Module):
    """Multi-modal Masked Autoencoder for pre-training."""
    

    def __init__(self, cfg, tracker: ExperimentTracker =None, preproc=None):
        super().__init__()
        self.cfg = cfg
        self.tracker = tracker
        self.modalities = cfg.modalities
        self.model = MultiMAEModel(self.cfg, tracker=tracker, preproc=preproc)

        self.decoder = MultiMAEDecoder(self.cfg, self.get_all_patches_number())
        if cfg.postprocessing:
            self.postprocessors = nn.ModuleDict({f"postprocessor_{modality}" : self.get_postprocessor(modality) for modality in self.modalities})
        self.contrastive_loss = None
        if self.cfg.contrastive_loss:
            print("CLIP loss: ")
            self.contrastive_loss =  CLIPAlignmentLoss()

        if "is_load_pretrained" in self.cfg and self.cfg.is_load_pretrained:
            print("cfg.pretrained_model_path: ", cfg.pretrained_model_path)
            model_state_dict = torch.load(cfg.pretrained_model_path)
            print("pretrained_params: ")
            for k in model_state_dict.keys():
                print(k)
            self.load_state_dict(model_state_dict)

    def get_postprocessor(self, modality):
        patch_size = self.get_patch_size(modality)
        if modality =="mri":
            return MriMAEDecoderPred(ViTMAEConfig(**self.cfg.mri_model))
        elif modality =="rna" or modality =="dnam" or modality =="wsi":
                return nn.Linear(
                    self.cfg.decoder_hidden_size, patch_size, bias=True
                )
        
    def get_patch_size(self,  modality: str)-> int:
        if modality =="rna" or modality=="dnam":
            if self.cfg.to_dict()[f"{modality}_model"]["patch_embedding"]["architecture"] == "tmae":
                    return self.cfg.to_dict()[f"{modality}_model"]["patch_size"]
            elif self.cfg.to_dict()[f"{modality}_model"]["patch_embedding"]["architecture"] == "clusterd_go":
                return self.model.encoders[modality].encoder.embeddings.patch_embeddings.max_cluster_lenth
        elif  modality =="mri":
            return self.cfg.to_dict()[f"{modality}_model"]["patch_size"] ** 3
        elif modality =="wsi":
            return (self.cfg.to_dict()[f"{modality}_model"]["patch_size"] ** 2) *3
        else:
            raise NotImplementedError(f"This modality - {modality} hasn't implemeted yet")
        
    def get_patches_number(self, modality: str) -> int:
        """Get number of patches for a specific modality.
        
        Args:
            modality: The modality type (e.g. "rna")
            
        Returns:
            Number of patches for the modality
        """
        return self.model.encoders[modality].encoder.embeddings.num_patches
     
    def get_all_patches_number(self) -> int:
        """Calculate total number of patches across all modalities.
        
        Returns:
            Total number of patches
        """
        num_patches = sum(self.get_patches_number(modality) for modality in self.modalities)
        return num_patches
    
    
    def __forward_loss(
        self,
        values: torch.FloatTensor,
        modality_mask: torch.FloatTensor,
        encoder: UnimodalEncoder,
        pred: torch.FloatTensor,
        mask: torch.FloatTensor,
        interpolate_pos_encoding: bool = False
    ) -> torch.FloatTensor:
        """Calculate reconstruction loss for a single modality.

        Args:
            values (torch.FloatTensor): Input tensor of shape (batch_size, num_channels, height, width)
                containing the original values.
            encoder (UnimodalEncoder): Encoder module for the current modality.
            pred (torch.FloatTensor): Predicted values tensor of shape 
                (batch_size, num_patches, patch_size**2 * num_channels).
            mask (torch.FloatTensor): Mask tensor of shape (batch_size, sequence_length)
                indicating which patches are masked (1) and which are not (0).
            interpolate_pos_encoding (bool, optional): Whether to interpolate position encodings.
                Defaults to False.

        Returns:
            torch.FloatTensor: Mean reconstruction loss on masked patches.
        """

        if encoder.modality == "wsi":
            values = values.squeeze(1)
  
        target = encoder.encoder.patchify(values, interpolate_pos_encoding=interpolate_pos_encoding)

        modality_mask = modality_mask.unsqueeze(1).to(mask.device)
        mask =  mask * modality_mask
        
        # Normalize target values if configured
        if self.cfg.norm_pix_loss:
            print("self.cfg.norm_pix_loss")
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()
    
        # Calculate mean squared error
        loss = (pred - target) ** 2
        # print("loss.shape", loss.shape)
        loss = loss.mean(dim=-1)  # Mean loss per patch [batch_size, num_patches]
        # print("loss.shape", loss.shape)
        # Calculate mean loss on masked patches only
       
        if (loss * mask).sum() >0:
            loss = (loss * mask).sum() / (mask.sum())
        else:
            loss = (loss * mask).sum()
        # print('Loss', loss)
        return loss
    
    def split_modalities(self, pred: torch.FloatTensor):
        start_idx = 0
        splitted_x = {}
        for modality in self.modalities:
            # Get number of patches for current modality
            num_patches = self.get_patches_number(modality)
            end_idx = start_idx + num_patches
            
            # Extract predictions and mask for current modality
            if self.cfg.postprocessing:
                splitted_x[modality] = self.postprocessors[f"postprocessor_{modality}"](pred[:, start_idx:end_idx])
            else:
                splitted_x[modality] = pred[:, start_idx:end_idx]
            start_idx = end_idx
        
        return splitted_x

    def forward_loss(self, x: dict[str, torch.FloatTensor],modality_masks: Dict[str, torch.FloatTensor], pred: torch.FloatTensor, mask: torch.FloatTensor, interpolate_pos_encoding: bool = False) -> torch.FloatTensor:
        """
        Calculate reconstruction loss across all modalities.

        Args:
            x (dict[str, torch.FloatTensor]): Dictionary mapping modality names to input tensors
                of shape (batch_size, num_channels, height, width).
            pred (torch.FloatTensor): Predicted values for all modalities concatenated,
                of shape (batch_size, total_patches, patch_size**2 * num_channels).
            mask (torch.FloatTensor): Mask indicating which patches are masked (1) and which are not (0),
                of shape (batch_size, total_patches).
            interpolate_pos_encoding (bool, optional): Whether to interpolate position encodings.
                Defaults to False.

        Returns:
            torch.FloatTensor: Total reconstruction loss summed across all modalities.
        """
        start_idx = 0
        total_loss = 0
        modality_losses = {}
        for modality in self.modalities:
            if modality =="clinical":
                if self.cfg.clinica_linear_last_fusion:
                    continue
            # if modality =="mri":
            #     modality_losses[modality] = torch.zeros(0).to(x[modality].device)
            #     continue
            if modality not in modality_losses:
                modality_losses[modality] = 0
            # Get number of patches for current modality
            num_patches = self.get_patches_number(modality)
            end_idx = start_idx + num_patches
            
            # Extract predictions and mask for current modality
            pred_modality = pred[:, start_idx:end_idx]
            mask_modality = mask[:, start_idx:end_idx]
 
            if self.cfg.postprocessing:
                pred_modality = self.postprocessors[f"postprocessor_{modality}"](pred_modality)


            # Calculate loss for current modality
            modality_loss = self.__forward_loss(
                x[modality],
                modality_masks[modality],
                self.model.encoders[modality], 
                pred_modality,
                mask_modality,
                interpolate_pos_encoding
            )


            modality_losses[modality] += modality_loss
            total_loss += modality_loss
            start_idx = end_idx

        return modality_losses, total_loss
    
   
    def contrastive_losses(self, latent: dict[str, torch.FloatTensor], modality_masks: Dict[str, torch.FloatTensor],):
        modalities_embeddings ={}   
        for modality in self.modalities:
            if modality =="clinical":
                if self.cfg.clinica_linear_last_fusion:
                    continue
        for modality in self.modalities:
            if modality =="clinical":
                if self.cfg.clinica_linear_last_fusion:
                    continue
            if modality not in modality_losses:
                modality_losses[modality] = 0
            # Get number of patches for current modality
            num_patches = self.get_patches_number(modality)
            end_idx = start_idx + num_patches
            
            # Extract predictions and mask for current modality
            latent_modality = latent[:, start_idx:end_idx]
            mask_modality = mask[:, start_idx:end_idx]
            
            modalities_embeddings[modality] = latent_modality

            total_loss += modality_loss
            start_idx = end_idx
        
        contrastive_coeff =0.2    
        contrastive_losses = {}     
        
        for modality in self.modalities:
            mask_modality = modality_masks[modality]
            if modality!="rna" and (torch.any(modality_masks[modality]) or torch.any(modality_masks[modality])):
                modalities_ids = torch.nonzero(mask_modality, as_tuple=True)[0].to(mask_modality.device)
                contrastive_loss =self.contrastive_loss.forward(modalities_embeddings["rna"][modalities_ids], modalities_embeddings[modality][modalities_ids])
                total_loss +=contrastive_loss
                contrastive_losses[f"rna-{modality}"]  = contrastive_loss
        return contrastive_losses        
                                
        
    def forward(
        self,
        x: Dict[str, torch.FloatTensor],
        modality_masks: Dict[str, torch.FloatTensor],
        interpolate_pos_encoding: bool = False
    ):
        """Forward pass through the model.
        
        Processes each modality separately, concatenates their embeddings,
        and passes through the decoder to get reconstructions.
        """
            
        return_attention = bool(getattr(self.cfg, "return_attention", False))
        concat_embedding = self.model(
            x,
            modality_masks,
            interpolate_pos_encoding,
            return_attention=return_attention,
        )

        latent = concat_embedding.last_hidden_state
        ids_restore = concat_embedding.ids_restore
        mask = concat_embedding.mask
        #Contrastive loss
        contrastive_losses ={}
        if self.contrastive_loss is not None:
            print("Contrastive loss: ")
            contrastive_losses = self.contrastive_losses(latent, modality_masks)
        # Decode the latent representations
        decoder_outputs = self.decoder(
            latent,
            ids_restore,
            output_attentions=return_attention,
            return_dict=True,
        )
        logits = decoder_outputs.logits

        # Calculate reconstruction loss
        modality_losses, total_loss = self.forward_loss(
            x,
            modality_masks,
            logits,
            mask,
            interpolate_pos_encoding=interpolate_pos_encoding
        )
            
        decoder_token_intervals: Dict[str, Tuple[int, int]] = {}
        token_start = 1  # 0 is decoder CLS
        for modality in self.modalities:
            token_end = token_start + self.get_patches_number(modality)
            decoder_token_intervals[modality] = (token_start, token_end)
            token_start = token_end

        attentions_payload = None
        if return_attention:
            encoder_fusion_attentions = concat_embedding.attentions
            if encoder_fusion_attentions is None:
                encoder_fusion_attentions = tuple()
            elif isinstance(encoder_fusion_attentions, torch.Tensor):
                encoder_fusion_attentions = (encoder_fusion_attentions,)
            else:
                encoder_fusion_attentions = tuple(encoder_fusion_attentions)

            decoder_attentions = decoder_outputs.attentions
            if decoder_attentions is None:
                decoder_attentions = tuple()
            elif isinstance(decoder_attentions, torch.Tensor):
                decoder_attentions = (decoder_attentions,)
            else:
                decoder_attentions = tuple(decoder_attentions)

            attentions_payload = {
                "encoder_fusion": encoder_fusion_attentions,
                "decoder": decoder_attentions,
                "token_intervals": {
                    "encoder_fusion": concat_embedding.modality_token_intervals or {},
                    "decoder": decoder_token_intervals,
                },
            }

        return ViTMAEForPreTrainingOutput(
            loss = (total_loss,modality_losses,contrastive_losses),
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=concat_embedding.hidden_states,
            attentions=attentions_payload,
        )
class MaskAttentionFusion(nn.Module):
    def __init__(self, fusion_depth, fusion_dim, fusion_nhead, fusion_dim_feedforward, fusion_dropout, activation_name ="relu"):
        super().__init__()
        self.fusion_depth = fusion_depth
        self.fusion_dim_feedforward = fusion_dim_feedforward
        
        if self.fusion_dim_feedforward > 0:
            self.fusion_layers = nn.ModuleList([nn.TransformerEncoderLayer(fusion_dim,
                                         fusion_nhead, dim_feedforward=fusion_dim_feedforward,
                                         dropout=fusion_dropout,activation =activation_name,
                                         batch_first=True, norm_first=True) for _ in range(fusion_depth)])
        else:    
            self.fusion_layers = nn.ModuleList([nn.MultiheadAttention(fusion_dim, fusion_nhead, dropout=fusion_dropout, batch_first=True) for _ in range(fusion_depth)])
        
    def _run_transformer_encoder_layer_with_attention(
        self,
        layer: nn.TransformerEncoderLayer,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        if not return_attention:
            return layer(x, src_key_padding_mask=key_padding_mask), None

        if layer.norm_first:
            x_norm = layer.norm1(x)
            attn_out, attn_weights = layer.self_attn(
                x_norm,
                x_norm,
                x_norm,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            x = x + layer.dropout1(attn_out)
            y = layer.norm2(x)
            y = layer.linear2(layer.dropout(layer.activation(layer.linear1(y))))
            x = x + layer.dropout2(y)
        else:
            attn_out, attn_weights = layer.self_attn(
                x,
                x,
                x,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            x = layer.norm1(x + layer.dropout1(attn_out))
            y = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(x + layer.dropout2(y))

        return x, attn_weights

    def forward(self, x, mask, return_attention: bool = False):
        all_layer_attentions: List[torch.Tensor] = []

        for layer in self.fusion_layers:
            if self.fusion_dim_feedforward > 0:
                # Keep existing behaviour: key padding mask is disabled in this branch.
                x, attn_weights = self._run_transformer_encoder_layer_with_attention(
                    layer,
                    x,
                    key_padding_mask=None,
                    return_attention=return_attention,
                )
            else:
                if return_attention:
                    x, attn_weights = layer(
                        x,
                        x,
                        x,
                        key_padding_mask=mask,
                        need_weights=True,
                        average_attn_weights=False,
                    )
                else:
                    x, _ = layer(x, x, x, key_padding_mask=mask)
                    attn_weights = None

            if return_attention and attn_weights is not None:
                all_layer_attentions.append(attn_weights)

        if return_attention:
            return x, tuple(all_layer_attentions)
        return x

class PerceiverMultiResampler(nn.Module):
     def __init__(self, fusion_depth, fusion_dim, modalities, modalities_dim, fusion_nhead):
         super().__init__()
         self.modalities = modalities
         self.perceivers = nn.ModuleDict({modality :  PerceiverResampler(
                                dim = fusion_dim,
                                depth = fusion_depth,
                                dim_head = modalities_dim[modality],
                                heads = fusion_nhead,
                                num_latents = modalities_dim[modality]) for modality in self.modalities})
         
     def forward(self, x,mask, modality_intervals):
            concat_x, concat_mask = [x[:,:1,:]], None
            if mask is not None:
                concat_mask = [mask[:,:1]]
            for modality in self.modalities:

                if mask is not None:
                    mask_modality =mask[:, modality_intervals[modality][0]:modality_intervals[modality][1]]
                x_perceived =self.perceivers[modality](x[:,modality_intervals[modality][0]:modality_intervals[modality][1],:])
                if mask is not None:
                    mask_modality = mask_modality[:,:x_perceived.shape[2]]
                    concat_mask.append(mask_modality)
                concat_x.append(torch.squeeze(x_perceived,1))
            if mask is not None:
                concat_mask = torch.cat(concat_mask, dim=1)       
            return torch.cat(concat_x, dim=1), mask
                       
class MultiMaeForSurvival(nn.Module):

    def __init__(self, cfg, tracker: ExperimentTracker = None, preproc=None):
        super().__init__()
        self.cfg = cfg
        self.tracker = tracker

        self.modalities = self.cfg.modalities
        
        if cfg.is_load_pretrained:
            model_state_dict = torch.load(cfg.pretrained_model_path)
            #  model_state_dict = {k.replace("model.", "", 1) if k.startswith("model.") else k: v 
            #                for k, v in model_state_dict.items()} 
            model_state_dict = {k.replace("model.", "", 1): v for k, v in model_state_dict.items() if k.startswith("model.")}     
            print("Load keys: ", model_state_dict.keys())

            self.model = MultiMAEModel(cfg, tracker= tracker, preproc=preproc)

            print("Model keys:", list(self.model.state_dict().keys()))
            self.model.load_state_dict(model_state_dict)
            if cfg.freezing_strategy:
                print("Freezing!")
                for name, param in self.model.named_parameters():
                    if ("cls_token" in name) or ("encoder_fusion_strategy" in name) or ("layer.11" in name) or ('encoders.dnam.encoder.encoder.layer.5' in name) or ('encoders.rna.encoder.encoder.layer.5' in name):
                        print("chozen", name)
                        param.requires_grad = True
                    else: 
                        param.requires_grad = False
                
        else:

            self.model = MultiMAEModel(cfg, tracker= tracker, preproc=preproc)

            
        if cfg.missing_modalities_strategy =="decoder":
            print("cfg.mm_pretrained_model_path: ", cfg.mm_pretrained_model_path)
            model_state_dict = torch.load(cfg.mm_pretrained_model_path)
            print("decoder_params: ", model_state_dict.keys())
            self.decoder_mm = MultiMaeForPretraining(ViTMAEConfig(**cfg.mm_decoder_config))
            self.decoder_mm.load_state_dict(model_state_dict)
            
            for param in self.decoder_mm.parameters():
                param.requires_grad = False
            # self.decoder_mm.eval()
            
        if cfg.fusion_strategy == "mask_attention":
            self.fusion_strategy = MaskAttentionFusion(cfg.fusion_depth, cfg.fusion_dim, cfg.fusion_nhead,
                                                       cfg.fusion_dim_feedforward, cfg.fusion_dropout)
            
        
        elif cfg.fusion_strategy == "perceived_attention":
            self.resampler = PerceiverResampler(
                                dim = cfg.perceiver_fusion_dim,
                                depth = cfg.perceiver_fusion_depth,
                                dim_head = cfg.perceiver_lattent_dim,
                                heads = cfg.perceiver_fusion_nhead,
                                num_latents = cfg.perceiver_lattent_dim)
            
            self.fusion_strategy = MaskAttentionFusion(cfg.fusion_depth, cfg.fusion_dim, cfg.fusion_nhead,
                                                       cfg.fusion_dim_feedforward, cfg.fusion_dropout)
        elif cfg.fusion_strategy == "perceived_modality_attention":
            self.resampler = PerceiverMultiResampler(cfg.perceiver_fusion_depth, cfg.perceiver_fusion_dim, 
                                                     self.modalities, cfg.perceiver_modalities_dim, cfg.perceiver_fusion_nhead) 
            
            self.fusion_strategy = MaskAttentionFusion(cfg.fusion_depth, cfg.fusion_dim, cfg.fusion_nhead,
                                                       cfg.fusion_dim_feedforward, cfg.fusion_dropout)          
        elif cfg.fusion_strategy =="linear":
            self.fusion_strategy = nn.Identity()
        
        elif self.cfg.fusion_strategy == "disentangled_fusion":
            self.fusion_strategy = MaskAttentionFusion(cfg.fusion_depth, cfg.fusion_dim, cfg.fusion_nhead,
                                                       cfg.fusion_dim_feedforward, cfg.fusion_dropout)
            number_modalities = len(self.modalities)
            if "clinical" in self.modalities:
                if self.cfg.clinica_linear_last_fusion:
                    number_modalities = number_modalities -1                                     
            self.agregation = nn.Linear(number_modalities+1, 1)

        else:
            raise ValueError(f"Invalid fusion strategy: {cfg.fusion_strategy}")
        
        if cfg.fusion_strategy_is_pretrained:
                model_state_dict = torch.load(cfg.pretrained_model_path)
                print(model_state_dict.keys())
                model_state_dict = {k.replace("fusion_strategy.", "", 1): v for k, v in model_state_dict.items() if k.startswith("fusion_strategy.")}  
                self.fusion_strategy.load_state_dict(model_state_dict)
                for param in self.fusion_strategy.parameters():
                    param.requires_grad = False
        # if "clinical" in self.modalities:
        #     #TODO add special param for it -> cfg.fusion_dim+3
        #     self.projection = nn.Linear(cfg.fusion_dim+3, cfg.output_dim)
        # else:           
        if self.cfg.multimodal_fusion =="single_head":
            if "clinical" in self.modalities and self.cfg.clinica_linear_last_fusion:
                #TODO add special param for it -> cfg.fusion_dim+3
                self.clinical_projection = nn.Sequential(nn.Linear(82, 32),
                                                    nn.LayerNorm(32),
                                                    nn.GELU(),
                                                    nn.Linear(32, 16),
                                                    nn.LayerNorm(16),
                                                    nn.GELU(),
                                                    nn.Dropout(0.5))
                self.projection = nn.Linear(cfg.fusion_dim+16, cfg.output_dim)
            else:
                self.projection = nn.Linear(cfg.fusion_dim, cfg.output_dim)
        else:
            self.projections =nn.ModuleDict({modality: nn.Linear(cfg.fusion_dim, cfg.output_dim) for modality in  cfg.modalities +["multimodal"]})           
        
        
    def get_primary_order(self, x,ids_restore):
        mask_tokens = self.model.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # unshuffle
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]).to(x_.device))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        return x

    def forward_unimodal_encoders(self, x: Dict[str, torch.FloatTensor], masks: Dict[str, torch.FloatTensor], interpolate_pos_encoding: bool = False):
        result = {}
        for modality_key, projection in self.projections.items():
                print("modality_key: ",modality_key, projection)
                if not modality_key =="multimodal":
                    x_modality = x[modality_key] 
                    embedding_modality = self.model.encoders[modality_key](x_modality)
                    print("random_patch_selection:", self.model.encoders["wsi"].encoder.config.random_patch_selection)
                    print("max_patches_per_sample", self.model.encoders["wsi"].encoder.config.max_patches_per_sample)
                    if modality_key =="wsi" and self.model.encoders["wsi"].encoder.config.max_patches_per_sample>1 and not self.model.encoders["wsi"].encoder.config.random_patch_selection:
                        print("Calculate mean")
                        n10, n_tokens, hidden_size = embedding_modality.last_hidden_state.shape
                        n = n10 // self.model.encoders["wsi"].encoder.config.max_patches_per_sample

                        last_hidden_state = embedding_modality.last_hidden_state.view(n, self.model.encoders["wsi"].encoder.config.max_patches_per_sample, n_tokens, hidden_size)  # (n, 10, n_tokens, hidden_size)
                        last_hidden_state = last_hidden_state.mean(dim=1)
                        result[modality_key] = projection(last_hidden_state[:,0,:])
                    else:
                        result[modality_key] = projection(embedding_modality.last_hidden_state[:,0,:])
                    print("result[modality_key]", result[modality_key].shape)
        return result        
            
    def forward(self, x: Dict[str, torch.FloatTensor], masks: Dict[str, torch.FloatTensor], interpolate_pos_encoding: bool = False):
        # print({modality: value.mean() for modality, value  in x.items()})
        if "clinical" in x.keys() and self.cfg.clinica_linear_last_fusion:

            clinical_data = x["clinical"].clone()
            del x["clinical"]
            del masks["clinical"]
            
        if self.cfg.missing_modalities_strategy =="decoder":
            decoded_x = self.decoder_mm(x, masks, interpolate_pos_encoding)
            decoded_x = self.decoder_mm.split_modalities(decoded_x.logits)
            for modality in self.modalities:
                if modality == "clinical":
                    continue
                if torch.any(~masks[modality]):
                    missing_modalities_ids = torch.nonzero(~masks[modality], as_tuple=True)[0].to(masks[modality].device)
                    ##  patchify todo
                    x[modality][missing_modalities_ids] = self.decoder_mm.model.encoders[modality].encoder.unpatchify(decoded_x[modality][missing_modalities_ids] , None)
                    masks[modality].fill_(1)

            
                
        concat_x = self.model(x, masks, interpolate_pos_encoding)

        if self.cfg.return_order:
            concat_x.last_hidden_state = self.get_primary_order(concat_x.last_hidden_state, concat_x.ids_restore)

        if self.cfg.fusion_masking:
            print("Cfg masking = None")
            concat_x.mask = None
            
        attn_weights = None

        if self.cfg.fusion_strategy == "perceived_attention":
            start_idx = 1
            intervals = dict()
            concat_x = self.resampler(concat_x.last_hidden_state)
            concat_x = self.fusion_strategy(torch.squeeze(concat_x,1), None)

        elif self.cfg.fusion_strategy == "perceived_modality_attention":
            start_idx = 1
            intervals = dict()
            for modality in self.modalities:
                num_patches = self.model.get_patches_number(modality)
                end_idx = start_idx + num_patches
                intervals[modality] = (start_idx,end_idx)
                start_idx = end_idx
            concat_x, mask  = self.resampler(concat_x.last_hidden_state,concat_x.mask, intervals)
            concat_x = self.fusion_strategy(torch.squeeze(concat_x,1), None)
               
        elif self.cfg.fusion_strategy == "mask_attention" and self.cfg.fusion_dim_feedforward==0:
            if self.cfg.return_attention:
                concat_x, attn_weights = self.fusion_strategy(
                    concat_x.last_hidden_state,
                    concat_x.mask,
                    return_attention=True,
                )
            else:
                concat_x = self.fusion_strategy(concat_x.last_hidden_state, concat_x.mask)
        elif self.cfg.fusion_strategy == "mask_attention" and self.cfg.fusion_dim_feedforward>0:
            if self.cfg.return_attention:
                concat_x, attn_weights = self.fusion_strategy(
                    concat_x.last_hidden_state,
                    concat_x.mask,
                    return_attention=True,
                )
            else:
                concat_x = self.fusion_strategy(concat_x.last_hidden_state, concat_x.mask)
        elif self.cfg.fusion_strategy == "disentangled_fusion":
            ## Get multimodal combination
            concat_x = self.fusion_strategy(concat_x.last_hidden_state, concat_x.mask)
            
            modalities_cls  =[]
            for modality in self.modalities:
                if modality=="clinical" and self.cfg.clinica_linear_last_fusion:
                    continue
                x_modality = x[modality] 
                embedding_modality = self.model.encoders[modality](x_modality)
                modalities_cls.append(embedding_modality.last_hidden_state[:,0,:])
            fused_embedding = torch.stack((concat_x[:,0,:], *modalities_cls), dim=1)
            fused_embedding = fused_embedding.permute(0,2,1)
       
            concat_x =self.agregation(fused_embedding)

            concat_x=concat_x.permute(0,2,1)
        else:
            concat_x = self.fusion_strategy(concat_x.last_hidden_state)
        
        
        
        # if "clinical" in self.modalities:
        #     clinical_logits = self.clinical_projection(clinical_data[:,0,:])
        #     concat_logits = torch.stack([logits,clinical_logits], dim =2)
        #     print("concat_logits.shape", concat_logits.shape)
        #     logits = self.final_projection(concat_logits)
        #     logits =torch.squeeze(logits,2)
        if self.cfg.multimodal_fusion =="single_head":
            if "clinical" in self.modalities and self.cfg.clinica_linear_last_fusion:

                clinical_logits = self.clinical_projection(clinical_data[:,0,:])    
                concat_with_clinical =torch.cat([concat_x[:,0,:], clinical_logits], axis =-1)
                logits = self.projection(concat_with_clinical)
                result = logits
            else:    
                logits = self.projection(concat_x[:,0,:])
                result = logits
        elif self.cfg.multimodal_fusion =="multiple_heads":
                result = self.projections["multimodal"](concat_x[:,0,:])
               
        if self.cfg.return_attention:
            return result, attn_weights
        else:
            return result
