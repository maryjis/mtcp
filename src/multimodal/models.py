import torch.nn as nn
import torch
from src.unimodal.rna.mae import RnaMAEModel, RnaMAEDecoder
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModelOutput, ViTMAEForPreTrainingOutput
from typing import Dict
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig

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
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size * config.num_channels, bias=True
        )  # encoder to decoder
        self.initialize_weights(num_patches)   
    
    
class MultiMaeForPretraining(nn.Module):
    """Multi-modal Masked Autoencoder for pre-training."""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.modalities = cfg.modalities
        self.__init_encoders__()
        self.encoders = nn.ModuleDict(self.encoders)
        self.decoder = MultiMAEDecoder(self.cfg, self.get_all_patches_number())
        
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
        loss = (loss * mask).sum() / mask.sum()
        
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
        
        for modality in self.modalities:
            # Get number of patches for current modality
            num_patches = self.get_patches_number(modality)
            end_idx = start_idx + num_patches
            
            # Extract predictions and mask for current modality
            pred_modality = pred[:, start_idx:end_idx]
            mask_modality = mask[:, start_idx:end_idx]
            
            # Calculate loss for current modality
            modality_loss = self.__forward_loss(
                x[modality],
                self.encoders[modality], 
                pred_modality,
                mask_modality,
                interpolate_pos_encoding
            )
            
            total_loss += modality_loss
            start_idx = end_idx
            
        return total_loss
    
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

        # Get indices of masked and unmasked samples
        empty_sample_ids = torch.nonzero(~mask, as_tuple=True)[0]
        non_empty_sample_ids = torch.nonzero(mask, as_tuple=True)[0]
        
        # Process non-empty samples
        non_empty_samples = torch.index_select(sample, dim=0, index=non_empty_sample_ids)
        embedded_non_empty = self.encoders[modality](non_empty_samples)
        
        # Create tensors for empty samples
        mask_empty = torch.zeros(batch_size, seq_length, device=device)
        ids_restore_empty = torch.arange(seq_length, device=device) + multimodal_lenths
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
        indices = torch.cat([empty_sample_ids, non_empty_sample_ids]).sort()[1]
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
            encoder_outputs.append(embedded_sample)
    
        # Combine embeddings from all modalities
        last_hidden_states = [out.last_hidden_state for out in encoder_outputs]
        masks = [out.mask for out in encoder_outputs]
        ids_restores = [out.ids_restore for out in encoder_outputs]
            
        concat_embedding = ViTMAEModelOutput(
            last_hidden_state=torch.cat(last_hidden_states, dim=1),
            mask=torch.cat(masks, dim=1),
            ids_restore=torch.cat(ids_restores, dim=1),
            hidden_states=None,
            attentions=None
        )    
            
        latent = concat_embedding.last_hidden_state
        ids_restore = concat_embedding.ids_restore
        mask = concat_embedding.mask

        # Decode the latent representations
        decoder_outputs = self.decoder(latent, ids_restore)
        logits = decoder_outputs.logits

        # Calculate reconstruction loss
        loss = self.forward_loss(
            x,
            logits,
            mask,
            interpolate_pos_encoding=interpolate_pos_encoding
        )
            
        return ViTMAEForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=concat_embedding.hidden_states,
            attentions=concat_embedding.attentions,
        )
