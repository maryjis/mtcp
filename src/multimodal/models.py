import torch.nn as nn
import torch
from src.unimodal.rna.mae import RnaMAEModel, RnaMAEDecoder
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModelOutput, ViTMAEDecoder, ViTMAEForPreTrainingOutput, ViTMAEDecoderOutput
from typing import Dict, List, Tuple, Union
from typing import Optional, Set
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from omegaconf import DictConfig, OmegaConf

class UnimodalEncoder(nn.Module):
    def __init__(self, encoder, unimodal_hidden_size, multimodal_hidden_size = None, is_projection = False):
        super().__init__()
        self.encoder = encoder
        self.inner_size = unimodal_hidden_size
        self.is_projection = is_projection
        if self.is_projection: 
            self.projection = nn.Linear(unimodal_hidden_size, multimodal_hidden_size)
            
    def forward(self, x):
        print(x.device)
        x = self.encoder(x)
        if self.is_projection:
               # TODO think about attention
               x.last_hidden_state = self.projection(x.last_hidden_state)
               #x.hidden_state = self.projection(x.hidden_state)
        return x 

class MultiMAEDecoder(RnaMAEDecoder):
    def __init__(self, config, num_patches):
        super().__init__(config, num_patches)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size * config.num_channels, bias=True
        )  # encoder to decoder
        self.initialize_weights(num_patches)
        
    def forward(
        self,
        hidden_states,
        ids_restore,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # embed tokens
        x = self.decoder_embed(hidden_states)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        hidden_states = x + self.decoder_pos_embed

        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    None,
                )
            else:
                layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)

        # remove cls token
        logits = logits[:, 1:, :]

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        return ViTMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )        
    
class MultiMaeForPretraining(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.modalities = cfg.modalities
        self.__init_encoders__()
        self.encoders =nn.ModuleDict(self.encoders)
        self.decoder = MultiMAEDecoder(self.cfg, self.get_all_patches_number())
        
    def __init_encoders__(self):
        self.encoders = {}
        for modality in self.modalities:
            if modality == "rna":
                cfg_rna_model = ViTMAEConfig(**self.cfg.rna_model)
                if cfg_rna_model.is_load_pretrained:
                    encoder = RnaMAEModel.from_pretrained(cfg_rna_model.pretrained_model_path, config = cfg_rna_model)
                else:
                    encoder = RnaMAEModel(cfg_rna_model) 
            self.encoders[modality] = UnimodalEncoder(encoder,cfg_rna_model.hidden_size, self.cfg.hidden_size, self.cfg.is_projection)

    def get_patches_number(self, modality):
            return self.encoders[modality].encoder.embeddings.num_patches
     
    def get_all_patches_number(self):
        print("Get all patches: ")
        num_patches = 0
        for modality in self.modalities:
            num_patches +=self.get_patches_number(modality)
            print("modality", num_patches)
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
        print("__forward_loss values.shape", values.shape)
        print("__forward_loss pred.shape", pred.shape)
        print("__forward_loss mask.shape", mask.shape)
        
        
        target = encoder.encoder.patchify(values, interpolate_pos_encoding=interpolate_pos_encoding)
        if self.cfg.norm_pix_loss:
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
            pred_modality, mask_modality = pred[:,prev_patches_number: prev_patches_number+modality_pathces_number], mask[:,prev_patches_number: prev_patches_number+modality_pathces_number]
            loss += self.__forward_loss(x[modality], self.encoders[modality], pred_modality, mask_modality, interpolate_pos_encoding)
            prev_patches_number += modality_pathces_number
        return loss    
        
    def forward(self, x: Dict[str, torch.FloatTensor] = None, masks:  Dict[str, torch.FloatTensor] = None, noise: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, interpolate_pos_encoding: bool = False):
                multimodal_lenths = 0
                encoder_outputs = []
                for i, modality in enumerate(self.modalities):
                    print("Modality: ", modality, " ", i)
                    seq_length = self.get_patches_number(modality)
                    sample = x[modality]
                    mask = masks[modality]
                    
                    batch_size, num_channels, sample_size = sample.shape
                    print("Sample.shape", sample.shape)
                    print("mask: ", mask)
                    if torch.all(mask):
                        print("mask: ")
                        embedded_sample = self.encoders[modality](
                                sample
                            )
                        print("embedded_sample.last_hidden_state.shape", embedded_sample.last_hidden_state.shape)
                        print("embedded_sample.mask.shape:", embedded_sample.mask.shape)
                        print("embedded_sample.ids_restore.shape:", embedded_sample.ids_restore.shape)
                        
                        print("Embedding ids_restore before changing:", embedded_sample.ids_restore)
                        embedded_sample.ids_restore = embedded_sample.ids_restore + multimodal_lenths
                        print("Embedding ids_restore after changing:", embedded_sample.ids_restore)
                        multimodal_lenths+=seq_length
                    else:
                        empty_sample_ids = torch.nonzero(mask == 0, as_tuple=True)
                        non_empty_sample_ids = torch.nonzero(mask == 1, as_tuple=True)
                        all_ids = torch.cat([empty_sample_ids, non_empty_sample_ids])
                        
                        not_empty_elements = torch.gather(sample, dim=0, index=non_empty_sample_ids)
                        
                        embedded_not_empty_elements = self.encoders[modality](
                                not_empty_elements
                            )
                        
                        
                        mask_empty_elements = torch.zeros(batch_size, seq_length, device=sample.device)
                        
                        
                        empty_ids_restore = torch.range(0,seq_length) + multimodal_lenths
                        
                        empty_sequence = self.mask_token.repeat(empty_sample_ids.shape[0], embedded_not_empty_elements.shape[1], 1)
                        
                        last_hidden_state = torch.cat([embedded_not_empty_elements.last_hidden_state, empty_sequence], axis = 0)
                        mask = torch.cat([embedded_not_empty_elements.mask, mask_empty_elements], axis = 0)
                        ids_restore =  torch.cat([embedded_not_empty_elements.ids_restore+multimodal_lenths, empty_ids_restore], axis = 0)
                        # hidden_states = torch.cat([embedded_not_empty_elements.hidden_states,
                        #                            torch.zeros(empty_sample_ids.shape[0],
                        #                                        embedded_not_empty_elements.hidden_states.shape[1], 
                        #                                        embedded_not_empty_elements.hidden_states.shape[2])])
                        # attentions = torch.cat([embedded_not_empty_elements.attentions,
                        #                            torch.zeros(empty_sample_ids.shape[0],
                        #                                        embedded_not_empty_elements.attentions.shape[1], 
                        #                                        embedded_not_empty_elements.attentions.shape[2])])
                        
                        last_hidden_state = torch.gather(last_hidden_state, dim=0, index=all_ids)
                        mask = torch.gather(mask, dim=0, index=all_ids)
                        ids_restore = torch.gather(ids_restore, dim=0, index=all_ids)
                        # hidden_states = torch.gather(hidden_states, dim=0, index=all_ids)
                        # attentions = torch.gather(attentions, dim=0, index=all_ids)
                        #last_hidden_state: torch.FloatTensor = None
                        # mask: torch.LongTensor = None
                        # ids_restore: torch.LongTensor = None
                        # hidden_states: Optional[Tuple[torch.FloatTensor]] = None
                        # attentions: Optional[Tuple[torch.FloatTensor]] = None
                        embedded_sample =  ViTMAEModelOutput(last_hidden_state = hidden_states,mask =mask,ids_restore = ids_restore,
                                                            hidden_states = None,attentions = None)
                        multimodal_lenths+=seq_length
                          
                    encoder_outputs.append(embedded_sample)
            
                last_hidden_states, masks,ids_restores, hidden_states, attentions  = [],[],[],[], []
                for encoder_output in encoder_outputs:
                    last_hidden_states.append(encoder_output.last_hidden_state)
                    masks.append(encoder_output.mask)
                    # hidden_states.append(encoder_output.hidden_states)
                    ids_restores.append(encoder_output.ids_restore)     
                    # attentions.append(attentions)  
                    
                concat_embedding =ViTMAEModelOutput(last_hidden_state = torch.cat(last_hidden_states,axis=1),mask =torch.cat(masks,axis=1),ids_restore = torch.cat(ids_restores, axis=1),
                                                                hidden_states = None, attentions = None)    
                    
                    
                latent = concat_embedding.last_hidden_state
                ids_restore = concat_embedding.ids_restore
                mask = concat_embedding.mask
                
                print("Latent.shape:", latent.shape)
                print("ids_restore.shape:", ids_restore.shape)
                print("mask.shape:", mask.shape)

                decoder_outputs = self.decoder(latent, ids_restore)
                    
                logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

                loss = self.forward_loss(x, logits, mask, interpolate_pos_encoding=interpolate_pos_encoding)
                    
                return ViTMAEForPreTrainingOutput(
                        loss=loss,
                        logits=logits,
                        mask=mask,
                        ids_restore=ids_restore,
                        hidden_states=concat_embedding.hidden_states,
                        attentions=concat_embedding.attentions,
                    )
