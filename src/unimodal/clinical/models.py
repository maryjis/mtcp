
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import numpy as np
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModel,ViTMAEDecoder, ViTMAEForPreTraining, get_1d_sincos_pos_embed_from_grid
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEModelOutput

# числовой эмбеддер
class NumericalEmbedder(nn.Module):
    def __init__(self, embed_dim, num_features):
        super().__init__()
        self.embedders = nn.ModuleList([
            nn.Sequential(nn.Linear(1, embed_dim), nn.LayerNorm(embed_dim))
            for _ in range(num_features)
        ])

    def forward(self, x):
        print(x[:, 1].unsqueeze(1))
        print(self.embedders[1](x[:, 1].unsqueeze(1)))
        tokens = [embed(x[:, i].unsqueeze(1)) for i, embed in enumerate(self.embedders)]
        tokens = torch.stack(tokens, dim=1)
        return tokens

# transformer блок
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x



# финальный класс
class ClinicalModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.cat_dims = self.config.cat_dims
        self.num_categories = len(self.cat_dims)
        self.num_unique_categories = sum(self.cat_dims)
        self.num_continuous = self.config.num_continuous
        self.total_features = self.num_categories + self.num_continuous
        self.embed_dim = self.config.embed_dim
        self.num_special_tokens = self.config.num_special_tokens
        self.total_tokens = self.num_unique_categories + self.num_special_tokens

        # categorical embedding
        if self.num_unique_categories > 0:
            self.categories_offset = F.pad(torch.tensor(list(self.cat_dims)), (1, 0), value = self.num_special_tokens)

            self.categories_offset = self.categories_offset.cumsum(dim = -1)[:-1]

        
            self.categorical_embeds = nn.Embedding(self.total_tokens, self.config.embed_dim)

        # numerical embedding
        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(self.embed_dim, self.num_continuous)

        # CLS токен
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # positional embedding (fixed sin-cos)
        pos = np.arange(self.total_features, dtype=np.float32)
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.embed_dim, pos)
        self.register_buffer('position_embeddings', torch.from_numpy(pos_embed).float().unsqueeze(0))  # [1, num_features, embed_dim]

        # transformer
        self.transformer_blocks = nn.ModuleList([nn.TransformerEncoderLayer(self.config.hidden_size, self.config.num_attention_heads,
                                         dim_feedforward=self.config.intermediate_size, 
                                         dropout=self.config.hidden_dropout_prob, activation=self.config.hidden_act,
                                   layer_norm_eps=1e-05, batch_first=True, norm_first=True) for _ in range(self.config.num_hidden_layers)])
                                   
    def split_numerical_and_categorical(self, x):

            x_cat =x[:,0,self.config.cat_indices]
            x_cont =x[:,0,self.config.cont_indices].float()
            return  x_cat.clone().long(), x_cont.clone()

    def forward(self, x):
        

        xs = []
        x_cat, x_cont = self.split_numerical_and_categorical(x)
        assert x_cat.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        if self.num_unique_categories > 0:

            x_cat = x_cat + self.categories_offset.to(x_cat.device)
            x_cat = self.categorical_embeds(x_cat)
            xs.append(x_cat)

        if self.num_continuous > 0:
            x_cont = self.numerical_embedder(x_cont)
            xs.append(x_cont)

        # объединяем токены
        x = torch.cat(xs, dim=1)

        # добавляем positional embeddings
        x = x + self.position_embeddings

        # CLS токен
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # transformer блоки
        for block in self.transformer_blocks:
            x = block(x)

   
        return ViTMAEModelOutput(last_hidden_state=x)

class ClinicalSurvivalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config =config
       
        self.model = ClinicalModel(config)
        self.projection = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.output_dim)
        )
        
    def forward(self, clinical_values, masks=None):
        x = self.model(clinical_values)
        x = self.projection(x.last_hidden_state[:,0,:])
        return x.squeeze(-1)