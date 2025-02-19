import torch.nn as nn
import torch
from src.unimodal.rna.mae import RnaMAEModel, RnaMAEDecoder
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModelOutput, ViTMAEForPreTrainingOutput
from typing import Dict
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from src.unimodal.mri.mae import MriMAEModel
from src.unimodal.dna.models import DNAmSurvivalModel, DNAmMAEModel
from src.unimodal.wsi.mae import WSIEmbeddingMAEModel

from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedModel
from src.unimodal.mri.mae import MriMAEDecoderPred
from flamingo_pytorch import PerceiverResampler
from src.multimodal.losses import CLIPAlignmentLoss

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
        
   
 
class MultiMAEModel(PreTrainedModel):
     
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.modalities = cfg.modalities
        self.__init_encoders__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.encoders = nn.ModuleDict(self.encoders)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.encoder_fusion_strategy = None
        if self.cfg.encoder_fusion_strategy =="masked_attention":
            self.encoder_fusion_strategy = MaskAttentionFusion(cfg.encoder_fusion_depth, cfg.encoder_fusion_dim,
                                                                cfg.encoder_fusion_nhead,
                                                        cfg.encoder_fusion_dim_feedforward, cfg.encoder_fusion_dropout)
        self.initialize_weights()

    def __init_encoders__(self):
        """Initialize encoders for each modality."""
        self.encoders = {}
        for modality in self.modalities:
            if modality == "rna":
                cfg_rna_model = ViTMAEConfig(**self.cfg.rna_model)
                encoder = None
                if cfg_rna_model.is_load_pretrained:
                    encoder = RnaMAEModel.from_pretrained(cfg_rna_model.pretrained_model_path, config=cfg_rna_model)
                    for param in encoder.parameters():
                        param.requires_grad = False
                else:
                    encoder = RnaMAEModel(cfg_rna_model)
                
                self.encoders[modality] = UnimodalEncoder(
                    encoder,
                    cfg_rna_model.hidden_size, 
                    self.cfg.hidden_size,
                    self.cfg.is_projection
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
                    self.cfg.is_projection
                )
            elif modality == "dnam":
                cfg_dnam_model = ViTMAEConfig(**self.cfg.dnam_model)
                encoder = None
                if cfg_dnam_model.is_load_pretrained:
                    encoder = DNAmMAEModel.from_pretrained(cfg_dnam_model.pretrained_model_path, config=cfg_dnam_model)
                    for param in encoder.parameters():
                        param.requires_grad = False
                else:
                    encoder = DNAmMAEModel(cfg_dnam_model)
                    
                self.encoders[modality] = UnimodalEncoder(
                    encoder,
                    cfg_dnam_model.hidden_size, 
                    self.cfg.hidden_size,
                    self.cfg.is_projection
                )
            elif modality == "wsi":
                cfg_wsi_model = ViTMAEConfig(**self.cfg.wsi_model)
                encoder = None
                if cfg_wsi_model.is_load_pretrained:
                    encoder = WSIEmbeddingMAEModel.from_pretrained(cfg_wsi_model.pretrained_model_path, config=cfg_wsi_model)
                    for param in encoder.parameters():
                        param.requires_grad = False
                else:
                    encoder = WSIEmbeddingMAEModel(config =cfg_wsi_model)
                    
                self.encoders[modality] = UnimodalEncoder(
                    encoder,
                    cfg_wsi_model.hidden_size, 
                    self.cfg.hidden_size,
                    self.cfg.is_projection
                )
            elif modality == "clinical":
                pass  
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
            mask_empty = torch.ones(len(empty_sample_ids), seq_length, device=device)
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
            mask_empty = torch.ones(len(empty_sample_ids), seq_length, device=device)
            ids_restore_empty = torch.arange(seq_length, device=device).repeat(len(empty_sample_ids), 1) + multimodal_lenths 
            len_keep = int(seq_length * (1 - self.encoders[modality].encoder.config.mask_ratio))
             
            empty_sequence = self.mask_token.repeat(len(empty_sample_ids),len_keep+1, 1)
            return ViTMAEModelOutput(
                last_hidden_state=empty_sequence,
                mask=mask_empty,
                ids_restore=ids_restore_empty,
                hidden_states=None,
                attentions=None
            )
            
    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=self.cfg.initializer_range)
        # torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)
        
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
            if modality =="clinical":
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
            
            embedded_sample = ViTMAEModelOutput(
                    last_hidden_state=embedded_sample.last_hidden_state[:,1:, :],
                    mask=embedded_sample.mask,
                    ids_restore=embedded_sample.ids_restore,
                    hidden_states=embedded_sample.hidden_states,
                    attentions=embedded_sample.attentions
                )
            encoder_outputs.append(embedded_sample)
      

        # Combine embeddings from all modalities
        last_hidden_states = torch.cat([out.last_hidden_state for out in encoder_outputs], dim=1)
        masks = torch.cat([out.mask for out in encoder_outputs], dim=1)
        ids_restores = torch.cat([out.ids_restore for out in encoder_outputs],dim=1)
        

        cls_tokens = self.cls_token.expand(last_hidden_states.shape[0], -1, -1)
        
        class_token_mask = torch.full((masks.shape[0], 1), 0).to(masks.device)
        masks = torch.cat((class_token_mask, masks), dim=1) 
        
        last_hidden_states = torch.cat((cls_tokens, last_hidden_states), dim=1)
        
        
        if self.encoder_fusion_strategy is not None:
            if last_hidden_states.shape[1] == masks.shape[1]:
                last_hidden_states = self.encoder_fusion_strategy(last_hidden_states, masks)
            else:
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
            self.postprocessors = nn.ModuleDict({f"postprocessor_{modality}" : self.get_postprocessor(modality) for modality in self.modalities})
        self.contrastive_loss = None
        if self.cfg.contrastive_loss:
            print("CLIP loss: ")
            self.contrastive_loss =  CLIPAlignmentLoss()  

    def get_postprocessor(self, modality):
        patch_size = self.get_patch_size(modality)
        if modality =="mri":
            return MriMAEDecoderPred(ViTMAEConfig(**self.cfg.mri_model))
        elif modality =="rna" or modality =="dnam" or modality =="wsi":
                return nn.Linear(
                    self.cfg.decoder_hidden_size, patch_size, bias=True
                )
        
    def get_patch_size(self,  modality: str)-> int:
        if modality =="rna" or modality=="dnam" or modality =="wsi":
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
        # Convert input to patches

        target = encoder.encoder.patchify(values, interpolate_pos_encoding=interpolate_pos_encoding)
        

        # Masked loss for all zero subjects (missing ones)
        modality_mask = modality_mask.unsqueeze(1).to(mask.device)
        mask =  mask * modality_mask
        print("mask.mean", mask.mean())
        print("target.mean", target.mean())
        print("pred.mean", pred.mean())
        
        # Normalize target values if configured
        if self.cfg.norm_pix_loss:
            print("self.cfg.norm_pix_loss")
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()
    
        # Calculate mean squared error
        loss = (pred - target).pow(2)

        loss = loss.mean(dim=-1)  # Mean loss per patch [batch_size, num_patches]

        # Calculate mean loss on masked patches only
  
        if (loss * mask).sum() >0:
            loss = (loss * mask).sum() / (mask.sum())
        else:
            print("Loss is zero!")
            loss = (loss * mask).sum() 
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
                continue
            if modality =="mri":
                modality_losses[modality] = torch.zeros(0).to(x[modality].device)
                continue
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
            print(f"modality {modality}, loss {modality_loss}")
            modality_losses[modality] += modality_loss
            total_loss += modality_loss
            start_idx = end_idx

        return modality_losses, total_loss
    
   
    def contrastive_losses(self, latent: dict[str, torch.FloatTensor], modality_masks: Dict[str, torch.FloatTensor],):
        modalities_embeddings ={}   
        for modality in self.modalities:
            if modality =="clinical":
                continue
        for modality in self.modalities:
            if modality =="clinical":
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
            print(mask_modality)
            if modality!="rna" and (torch.any(modality_masks[modality]) or torch.any(modality_masks[modality])):
                modalities_ids = torch.nonzero(mask_modality, as_tuple=True)[0].to(mask_modality.device)
                print(modalities_ids)
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
            
        concat_embedding = self.model(x, modality_masks, interpolate_pos_encoding)
            
        latent = concat_embedding.last_hidden_state
        ids_restore = concat_embedding.ids_restore
        mask = concat_embedding.mask
        #Contrastive loss
        contrastive_losses ={}
        if self.contrastive_loss is not None:
            print("Contrastive loss: ")
            contrastive_losses = self.contrastive_losses(latent, modality_masks)
        # Decode the latent representations
        decoder_outputs = self.decoder(latent, ids_restore)
        logits = decoder_outputs.logits

        # Calculate reconstruction loss
        modality_losses, total_loss = self.forward_loss(
            x,
            modality_masks,
            logits,
            mask,
            interpolate_pos_encoding=interpolate_pos_encoding
        )
            
        return ViTMAEForPreTrainingOutput(
            loss = (total_loss,modality_losses,contrastive_losses),
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
                x = layer(x, src_key_padding_mask= mask if mask is not None else None)
            else:
               x = layer(x,x,x, key_padding_mask  = mask if mask is not None else None)[0]           
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
                print(modality_intervals[modality])
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
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.modalities = self.cfg.modalities
        
        if cfg.is_load_pretrained:
            model_state_dict = torch.load(cfg.pretrained_model_path)
            #  model_state_dict = {k.replace("model.", "", 1) if k.startswith("model.") else k: v 
            #                for k, v in model_state_dict.items()} 
            model_state_dict = {k.replace("model.", "", 1): v for k, v in model_state_dict.items() if k.startswith("model.")}     
            print("Load keys: ", model_state_dict.keys())
            self.model = MultiMAEModel(cfg)
            print("Model keys:", list(self.model.state_dict().keys()))
            self.model.load_state_dict(model_state_dict)
            if cfg.freezing_strategy:
                print("Freezing!")
                for param in self.model.parameters():
                    param.requires_grad = False
                
        else:
            self.model = MultiMAEModel(cfg)
            
        if cfg.missing_modalities_strategy =="decoder":
            model_state_dict = torch.load(cfg.mm_pretrained_model_path)
            self.decoder_mm = MultiMaeForPretraining(ViTMAEConfig(**cfg.mm_decoder_config))
            self.decoder_mm.load_state_dict(model_state_dict)
            
            for param in self.decoder_mm.parameters():
                param.requires_grad = False
            
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
            self.agregation = nn.Linear(len(self.modalities)+1, 1)
            
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
        
        if "clinical" in self.modalities:
            #TODO add special param for it -> cfg.fusion_dim+3
            self.clinical_projection = nn.Linear(3, 3)
            self.projection = nn.Linear(cfg.fusion_dim+3, cfg.output_dim)
        else:
            self.projection = nn.Linear(cfg.fusion_dim, cfg.output_dim)    
        
        
    def get_primary_order(self, x,ids_restore):
        mask_tokens = self.model.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # unshuffle
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]).to(x_.device))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        return x
            
            
    def forward(self, x: Dict[str, torch.FloatTensor], masks: Dict[str, torch.FloatTensor], interpolate_pos_encoding: bool = False):
        # print({modality: value.mean() for modality, value  in x.items()})
        if "clinical" in x.keys():

            clinical_data = x["clinical"].clone()
            del x["clinical"]
            del masks["clinical"]
            
        if self.cfg.missing_modalities_strategy =="decoder":
            self.decoder_mm.eval()
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
               
        elif self.cfg.fusion_strategy == "mask_attention":
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
        
        print(self.modalities)
        if "clinical" in self.modalities:
            clinical_logits = self.clinical_projection(clinical_data[:,0,:])    
            concat_with_clinical =torch.cat([concat_x[:,0,:], clinical_logits], axis =-1)
            print("concat_with_clinical.shape ", concat_with_clinical.shape)
            logits = self.projection(concat_with_clinical) 
        else:    
            logits = self.projection(concat_x[:,0,:])
        # if "clinical" in self.modalities:
        #     clinical_logits = self.clinical_projection(clinical_data[:,0,:])
        #     concat_logits = torch.stack([logits,clinical_logits], dim =2)
        #     print("concat_logits.shape", concat_logits.shape)
        #     logits = self.final_projection(concat_logits)
        #     logits =torch.squeeze(logits,2)
        return logits