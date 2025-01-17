import torch.nn as nn
import torch
from src.unimodal.rna.mae import RnaMAEModel, RnaMAEDecoder
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModelOutput, ViTMAEForPreTrainingOutput
from typing import Dict
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from src.unimodal.mri.mae import MriMAEModel
from src.unimodal.dna.models import DNAmSurvivalModel, DNAmMAEModel
from omegaconf import DictConfig, OmegaConf

class UnimodalEncoder(nn.Module):
    def __init__(self, encoder, unimodal_hidden_size, multimodal_hidden_size = None, is_projection = False):
        super().__init__()
        self.encoder = encoder
        self.inner_size = unimodal_hidden_size
        self.is_projection = is_projection
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
        
   
 
class MultiMAEModel(nn.Module):
     
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.modalities = cfg.modalities
        self.__init_encoders__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.encoders = nn.ModuleDict(self.encoders)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        if cfg.encoder_fusion_strategy == "mask_attention":
            self.encoder_fusion_strategy = MaskAttentionFusion(cfg.encoder_fusion_depth, cfg.encoder_fusion_dim,
                                                               cfg.encoder_fusion_nhead,
                                                       cfg.encoder_fusion_dim_feedforward, cfg.encoder_fusion_dropout)
        else:
            self.encoder_fusion_strategy = nn.Identity()
        self.initialize_weights()

    def __init_encoders__(self):
        """Initialize encoders for each modality."""
        self.encoders = {}
        for modality in self.modalities:
            if modality == "rna":
                cfg_rna_model = ViTMAEConfig(**self.cfg.rna_model)
                encoder = (
                    RnaMAEModel.from_pretrained(cfg_rna_model.pretrained_model_path, config=cfg_rna_model)
                    if cfg_rna_model.is_load_pretrained
                    else RnaMAEModel(cfg_rna_model)
                )
                self.encoders[modality] = UnimodalEncoder(
                    encoder,
                    cfg_rna_model.hidden_size, 
                    self.cfg.hidden_size,
                    self.cfg.is_projection
                )
            elif modality == "mri":
                cfg_mri_model = ViTMAEConfig(**self.cfg.mri_model)
                self.encoders[modality] = UnimodalEncoder(
                    MriMAEModel.from_pretrained(cfg_mri_model.pretrained_model_path, config=cfg_mri_model)
                    if cfg_mri_model.is_load_pretrained
                    else MriMAEModel(cfg_mri_model),
                    cfg_mri_model.hidden_size, 
                    self.cfg.hidden_size,
                    self.cfg.is_projection
                )
            elif modality == "dnam":
                cfg_dnam_model = ViTMAEConfig(**self.cfg.dnam_model)
                self.encoders[modality] = UnimodalEncoder(
                    DNAmMAEModel.from_pretrained(cfg_dnam_model.pretrained_model_path, config=cfg_dnam_model)
                    if cfg_dnam_model.is_load_pretrained
                    else DNAmMAEModel(cfg_dnam_model),
                    cfg_dnam_model.hidden_size, 
                    self.cfg.hidden_size,
                    self.cfg.is_projection
                )
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
        return self.encoders[modality].encoder.embeddings.num_patches
     
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
        # print("modality: ", modality)
        # print("Seq length: ", seq_length)
        # print(mask)
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
            mask_empty = torch.zeros(len(empty_sample_ids), seq_length, device=device)
            ids_restore_empty = torch.arange(seq_length, device=device).repeat(len(empty_sample_ids), 1) + multimodal_lenths  
                      
            non_empty_sample_ids = torch.nonzero(mask, as_tuple=True)[0].to(device)
            # print("Non empty sample ids: ", non_empty_sample_ids)
            
            # Process non-empty samples
            non_empty_samples = torch.index_select(sample, dim=0, index=non_empty_sample_ids)
            # print("Non empty samples shape: ", non_empty_samples.shape)
            embedded_non_empty = self.encoders[modality](non_empty_samples)
            
            empty_sequence = self.mask_token.repeat(len(empty_sample_ids), embedded_non_empty.last_hidden_state.shape[1], 1)
            # print("Empty sample ids: ", empty_sample_ids)
            # print("Non empty sequence shape: ", embedded_non_empty.last_hidden_state.shape)
            # print("Empty sequence shape: ", empty_sequence.shape)
            # Combine empty and non-empty tensors
            last_hidden_state = torch.cat([
                embedded_non_empty.last_hidden_state,
                empty_sequence
            ], dim=0)
            
            mask_combined = torch.cat([
                embedded_non_empty.mask,
                mask_empty
            ], dim=0)
            # print("Ids restore empty shape: ", ids_restore_empty.shape)
            # print("embedded_non_empty.ids_restore: ", embedded_non_empty.ids_restore.shape)
            ids_restore = torch.cat([
                embedded_non_empty.ids_restore + multimodal_lenths,
                ids_restore_empty
            ], dim=0)

            # Reorder tensors based on original sample ordering
            #indices = torch.cat([empty_sample_ids, non_empty_sample_ids]).sort()[1]
            indices = torch.argsort(torch.cat([non_empty_sample_ids, empty_sample_ids]), dim=0)
            
            # print("Last hidden state shape: ", last_hidden_state.shape)
            # print("Mask combined shape: ", mask_combined.shape)
            # print("Mask combined: ", torch.any(mask_combined, dim=1))
            
            # print("Indices: ", indices)
            last_hidden_state = torch.index_select(last_hidden_state, dim=0, index=indices)
            mask_combined = torch.index_select(mask_combined, dim=0, index=indices)
            ids_restore = torch.index_select(ids_restore, dim=0, index=indices)
            
            # print("Last hidden state shape: ", last_hidden_state.shape)
            # print("Mask combined shape: ", mask_combined.shape)
            # print("Mask combined: ", torch.any(mask_combined, dim=1))
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
            mask_empty = torch.zeros(len(empty_sample_ids), seq_length, device=device)
            ids_restore_empty = torch.arange(seq_length, device=device).repeat(len(empty_sample_ids), 1) + multimodal_lenths 
            len_keep = int(seq_length * (1 - self.cfg.mask_ratio))
             
            empty_sequence = self.mask_token.repeat(len(empty_sample_ids), len_keep+1, 1)
            return ViTMAEModelOutput(
                last_hidden_state=empty_sequence,
                mask=mask_empty,
                ids_restore=ids_restore_empty,
                hidden_states=None,
                attentions=None
            )
            
    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=self.cfg.initializer_range)
        
    def forward(
        self,
        x: Dict[str, torch.FloatTensor],
        masks: Dict[str, torch.FloatTensor],
        interpolate_pos_encoding: bool = False
    ):
        """Forward pass through the model.
        
        Processes each modality separately, concatenates their embeddings,
        and passes through the decoder to get reconstructions.
        """
        # Track cumulative sequence length across modalities
        multimodal_length = 0
        encoder_outputs = []

        # Process each modality
        is_first = True

        for modality in self.modalities:
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
            
            embedded_sample = ViTMAEModelOutput(
                    last_hidden_state=embedded_sample.last_hidden_state[:,1:, :],
                    mask=embedded_sample.mask,
                    ids_restore=embedded_sample.ids_restore,
                    hidden_states=embedded_sample.hidden_states,
                    attentions=embedded_sample.attentions
                )
            encoder_outputs.append(embedded_sample)
            print(f"{modality} embed shape:", embedded_sample.last_hidden_state.shape)           

        # Combine embeddings from all modalities
        last_hidden_states = torch.cat([out.last_hidden_state for out in encoder_outputs], dim=1)
        masks = torch.cat([out.mask for out in encoder_outputs], dim=1)
        ids_restores = torch.cat([out.ids_restore for out in encoder_outputs],dim=1)
        
        print("last_hidden_states.shape: ", last_hidden_states.shape)  
        print("masks.shape: ", masks.shape)  
        cls_tokens = self.cls_token.expand(last_hidden_states.shape[0], -1, -1)
        
        class_token_mask = torch.full((masks.shape[0], 1), 1).to(masks.device)
        masks = torch.cat((class_token_mask, masks), dim=1) 
        
        last_hidden_states = torch.cat((cls_tokens, last_hidden_states), dim=1)
        
        print("after preproc last_hidden_states.shape: ", last_hidden_states.shape)  
        print("after preproc masks.shape: ", masks.shape)  
        
        last_hidden_states = self.encoder_fusion_strategy(last_hidden_states, None) 
         
        concat_embedding = ViTMAEModelOutput(
            last_hidden_state=last_hidden_states,
            mask=masks,
            ids_restore=ids_restores,
            hidden_states=None,
            attentions=None
        )    
            
        return concat_embedding
        
        
class MultiMaeForPretraining(nn.Module):
    """Multi-modal Masked Autoencoder for pre-training."""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.modalities = cfg.modalities
        self.model = MultiMAEModel(self.cfg)
        self.decoder = MultiMAEDecoder(self.cfg, self.get_all_patches_number())
        if cfg.postprocessing:
            self.postprocessors = nn.ModuleDict({f"postprocessor_{modality}" : nn.Conv1d(self.cfg.decoder_hidden_size,
                                                                                         self.get_patch_size(modality), kernel_size =1, stride=1, padding=0) for modality in self.modalities})
        

    def get_patch_size(self,  modality: str)-> int:
        if modality =="rna" or modality=="dnam":
            return self.cfg.to_dict()[f"{modality}_model"]["patch_size"]
        elif  modality =="mri":
            return self.cfg.to_dict()[f"{modality}_model"]["patch_size"] ** 3
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
        # Convert input to patches
        target = encoder.encoder.patchify(values, interpolate_pos_encoding=interpolate_pos_encoding)

        # Normalize target values if configured
        if self.cfg.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()

        # Calculate mean squared error

        loss = (pred - target).pow(2)

        loss = loss.mean(dim=-1)  # Mean loss per patch [batch_size, num_patches]

        # Calculate mean loss on masked patches only
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        
        return loss
    
    def forward_loss(self, x: dict[str, torch.FloatTensor], pred: torch.FloatTensor, mask: torch.FloatTensor, interpolate_pos_encoding: bool = False) -> torch.FloatTensor:
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
            if modality not in modality_losses:
                modality_losses[modality] = 0
            # Get number of patches for current modality
            num_patches = self.get_patches_number(modality)
            end_idx = start_idx + num_patches
            
            # Extract predictions and mask for current modality
            pred_modality = pred[:, start_idx:end_idx]
            mask_modality = mask[:, start_idx:end_idx]
            if self.cfg.postprocessing:
                pred_modality = self.postprocessors[f"postprocessor_{modality}"](pred_modality.permute(0, 2, 1))
                pred_modality = pred_modality.permute(0, 2, 1)
    
            # Calculate loss for current modality
            modality_loss = self.__forward_loss(
                x[modality],
                self.model.encoders[modality], 
                pred_modality,
                mask_modality,
                interpolate_pos_encoding
            )

            modality_losses[modality] += modality_loss
            total_loss += modality_loss
            start_idx = end_idx
            
        return modality_losses, total_loss
    
   
                             
        
    def forward(
        self,
        x: Dict[str, torch.FloatTensor],
        masks: Dict[str, torch.FloatTensor],
        interpolate_pos_encoding: bool = False
    ):
        """Forward pass through the model.
        
        Processes each modality separately, concatenates their embeddings,
        and passes through the decoder to get reconstructions.
        """
            
        concat_embedding = self.model(x, masks, interpolate_pos_encoding)
            
        latent = concat_embedding.last_hidden_state
        ids_restore = concat_embedding.ids_restore
        mask = concat_embedding.mask

        # Decode the latent representations
        decoder_outputs = self.decoder(latent, ids_restore)
        logits = decoder_outputs.logits

        # Calculate reconstruction loss
        modality_losses, total_loss = self.forward_loss(
            x,
            logits,
            mask,
            interpolate_pos_encoding=interpolate_pos_encoding
        )
            
        return ViTMAEForPreTrainingOutput(
            loss = (total_loss,modality_losses),
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=concat_embedding.hidden_states,
            attentions=concat_embedding.attentions,
        )
class MaskAttentionFusion(nn.Module):
    def __init__(self, fusion_depth, fusion_dim, fusion_nhead, fusion_dim_feedforward, fusion_dropout):
        super().__init__()
        self.fusion_depth = fusion_depth
        self.fusion_dim_feedforward = fusion_dim_feedforward
        if self.fusion_dim_feedforward > 0:
            self.fusion_layers = nn.ModuleList([nn.TransformerEncoderLayer(fusion_dim, fusion_nhead, dim_feedforward=fusion_dim_feedforward, dropout=fusion_dropout,
                                   layer_norm_eps=1e-05, batch_first=True) for _ in range(fusion_depth)])
        else:    
            self.fusion_layers = nn.ModuleList([nn.MultiheadAttention(fusion_dim, fusion_nhead, dropout=fusion_dropout, batch_first=True) for _ in range(fusion_depth)])
        
    def forward(self, x, mask):
        
        for layer in self.fusion_layers:
            if self.fusion_dim_feedforward > 0:
                x = layer(x, src_key_padding_mask= 1 - mask if mask is not None else None)
            else:
               x = layer(x,x,x, key_padding_mask  = 1 - mask if mask is not None else None)[0]           
        return x
        
class MultiMaeForSurvival(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.modalities = self.cfg.modalities
        
        if cfg.is_load_pretrained:
            self.model = MultiMAEModel.from_pretrained(cfg.pretrained_model_path, config=cfg)
        else:
            self.model = MultiMAEModel(cfg)
        if cfg.fusion_strategy == "mask_attention":
            self.fusion_strategy = MaskAttentionFusion(cfg.fusion_depth, cfg.fusion_dim, cfg.fusion_nhead,
                                                       cfg.fusion_dim_feedforward, cfg.fusion_dropout)
        elif cfg.fusion_strategy =="linear":
            self.fusion_strategy = nn.Identity()
        
        elif self.cfg.fusion_strategy == "disentangled_fusion":
            self.fusion_strategy = MaskAttentionFusion(cfg.fusion_depth, cfg.fusion_dim, cfg.fusion_nhead,
                                                       cfg.fusion_dim_feedforward, cfg.fusion_dropout)
            self.agregation = nn.Linear(len(self.modalities)+1, 1)
            
        else:
            raise ValueError(f"Invalid fusion strategy: {cfg.fusion_strategy}")
            
        self.projection = nn.Linear(cfg.fusion_dim, cfg.output_dim)
        
        
            
    def forward(self, x: Dict[str, torch.FloatTensor], masks: Dict[str, torch.FloatTensor], interpolate_pos_encoding: bool = False):
        # print({modality: value.mean() for modality, value  in x.items()})
        concat_x = self.model(x, masks, interpolate_pos_encoding)
        # print("Mask shape: ", x.mask.shape)
        # print("Last_hidden_state shape: ", x.last_hidden_state.shape)

        if self.cfg.fusion_strategy == "mask_attention":
            concat_x = self.fusion_strategy(concat_x.last_hidden_state, concat_x.mask)
        elif self.cfg.fusion_strategy == "disentangled_fusion":
            ## Get multimodal combination
            concat_x = self.fusion_strategy(concat_x.last_hidden_state, concat_x.mask)
            
            modalities_cls  =[]
            for modality in self.modalities:
                x_modality = x[modality] 
                embedding_modality = self.model.encoders[modality](x_modality)
                modalities_cls.append(embedding_modality.last_hidden_state[:,0,:])
            fused_embedding = torch.stack((concat_x[:,0,:], *modalities_cls), dim=1)
            print(fused_embedding.shape)
            fused_embedding = fused_embedding.permute(0,2,1)
            concat_x =self.agregation(fused_embedding).permute(0,2,1)
            print(concat_x.shape)
        else:
            concat_x = self.fusion_strategy(concat_x.last_hidden_state)
        logits = self.projection(concat_x[:,0,:])
        return logits