import torch.nn as nn
import torch
from src.unimodal.rna.mae import RnaMAEModel
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModelOutput, ViTMAEDecoder, ViTMAEForPreTraining
from typing import Dict, List, Tuple, Union

class UnimodalEncoder(nn.Module):
    def __init__(encoder, unimodal_hidden_size, multimodal_hidden_size = None, is_projection = False):
        self.encoder = encoder
        self.inner_size = unimodal_hidden_size
        self.is_projection = is_projection
        if self.is_projection: 
            self.projection = nn.Linear(unimodal_hidden_size, multimodal_hidden_size)
            
   def forward(x):
        x = self.encoder(x)
        if self.is_projection:
               # TODO think about attention
               x.last_hidden_state = self.projection(x.last_hidden_state)
               x.hidden_state = self.projection(x.hidden_state)
        return x 

class MultiMAEDecoder(ViTMAEDecoder):
    def __init__(self, config, num_patches):
        super().__init__(config, num_patches)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size * config.num_channels, bias=True
        )  # encoder to decoder
        self.initialize_weights(num_patches)        
    
class MultiMaeForPretraining(ViTMAEForPreTraining):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.modalities = cfg.modalities
        self.__init_encoders__()
        self.decoder = ViTMAEDecoder(self.cfg, self.get_all_patches_number())
        
    def __init_encoders__(self):
        self.encoders = {}
        for modality in self.modalities:
            if modality == "rna":
                if config.rna_model.is_load_pretrained:
                    encoder = RnaMAEModel.from_pretrained(self.cfg.rna_model.pretrained_model_path, config = self.cfg.rna_model)
                else:
                    encoder = RnaMAEModel(self.cfg.rna_model) 
            self.encoders[modality] = UnimodalEncoder(encoder,self.cfg.rna_model.hidden_size, self.cfg.hidden_size, self.cfg.is_projection)

    def get_patches_number(self,modality):
            return self.encoders[modality].embeddings.num_patches
     
    def get_all_patches_number(self):
        num_patches = 0
        for modality in self.modalities:
            num_patches +=self.get_all_patches_number(modality)
        return num_patches
    
    
    def __forward_loss(self, values, encoder, pred, mask, interpolate_pos_encoding: bool = False):
        """
        Args:
            x (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Multimodal  values.
            pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Predicted pixel values.
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).
            interpolate_pos_encoding (`bool`, *optional*, default `False`):
                interpolation flag passed during the forward pass.

        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """
        target = encoder.patchify(values, interpolate_pos_encoding=interpolate_pos_encoding)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def forward_loss(self, x, pred, mask, interpolate_pos_encoding: bool = False):
        """
        Args:
            x (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Multimodal  values.
            pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Predicted pixel values.
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).
            interpolate_pos_encoding (`bool`, *optional*, default `False`):
                interpolation flag passed during the forward pass.

        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """
        prev_patches_number = 0
        loss = 0
        for i, modality in enumerate(self.modalities):
            modality_pathces_number = self.get_patches_number(modality)
            pred, mask = pred[prev_patches_number: modality_pathces_number], mask[prev_patches_number: modality_pathces_number]
            loss += __forward_loss(self, x[modality], self.encoders['modality'], pred, mask, interpolate_pos_encoding: bool = False)
            prev_patches_number += modality_pathces_number
        return loss    
        
    def forward(x: Dict[str, torch.FloatTensor] = None,
                noise: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                interpolate_pos_encoding: bool = False):
        multimodal_lenths = 0
        encoder_outputs = []
        for i, modality in enumerate(self.modalities):
            sample, mask =x[modality]
            if mask:
                embedded_sample = self.encoders[modality](
                        sample,
                        noise=noise,
                        head_mask=head_mask,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        interpolate_pos_encoding=interpolate_pos_encoding,
                    )

                embedded_sample.ids_restore = embedded_sample.ids_restore + multimodal_lenths
                multimodal_lenths+=embedded_sample.mask.shape[0]
            else:
                num_patches = self.get_patches_number(modality, sample.shape[-1])
                ids_restore = torch.range(0,num_patches) + multimodal_lenths
                
                #last_hidden_state: torch.FloatTensor = None
                # mask: torch.LongTensor = None
                # ids_restore: torch.LongTensor = None
                # hidden_states: Optional[Tuple[torch.FloatTensor]] = None
                # attentions: Optional[Tuple[torch.FloatTensor]] = None
                embedded_sample =  ViTMAEModelOutput(last_hidden_state = torch.tensor([]),mask =torch.zeros(num_patches),ids_restore = ids_restore
                                                     hidden_states = torch.tensor([]),attentions = torch.tensor([]))
            encoder_outputs.append(embedded_sample)
       
       last_hidden_state, mask,ids_restore, hidden_states, attentions  = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
       for encoder_output in encoder_outputs:
           last_hidden_state = torch.concat([last_hidden_state, encoder_output.last_hidden_state], axis =1)
           mask = torch.concat([mask, encoder_output.mask])
           hidden_states = torch.concat([hidden_states, encoder_output.hidden_states], axis =1)
           ids_restore = torch.concat([ids_restore, encoder_output.ids_restore])     
           attentions =   torch.concat([attentions, encoder_output.attentions])  
        
        concat_embedding =ViTMAEModelOutput(last_hidden_state = last_hidden_state, mask = mask,ids_restore = ids_restore
                                                     hidden_states = hidden_states, attentions = attentions)    
        
        
        latent = concat_embedding.last_hidden_state
        ids_restore = concat_embedding.ids_restore
        mask = concat_embedding.mask

        decoder_outputs = self.decoder(latent, ids_restore, interpolate_pos_encoding=interpolate_pos_encoding)
        
        logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        loss = self.forward_loss(x, logits, mask, interpolate_pos_encoding=interpolate_pos_encoding)
        
        if not return_dict:
            output = (logits, mask, ids_restore) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ViTMAEForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=concat_embedding.hidden_states,
            attentions=concat_embedding.attentions,
        )
