import torch
from torch import nn

#same as paper code
d_model=768,
nhead=6,
num_layers=6,
dim_feedforward=3072,
dropout=0.1,
activation=nn.GELU(),
layer_norm_eps=1e-05,
batch_first=True,
norm_first=True,
context_length=1024,

class Numformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #code uses Encoder but in the paper the model is said to have the main features as GPT-2 so I used a Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first)
        
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers)
        
        #text needs to be embedded and given the positional information
        self.token_embed_table = nn.Embedding(vocab_size, d_model)
        self.position_embed_table = nn.Embedding(context_length, d_model)
        
        #token head and num head with hidden layer dimension = 4*embedding dimension
        self.lm_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation,
            nn.Linear(dim_feedforward, vocab_size)
        )
        self.num_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation,
            nn.Linear(dim_feedforward, 1)
        )


    def forward(self, x_text, x_num):
        batch_dim, time_dim = x.shape

        tok_emb = self.token_embed_table(x_text)
        #pos_emb takes as input a tensor with values from 0 to time_dim-1 (time_dim == context_length)
        pos_emb = self.position_embed_table(torch.arange(time_dim))

        x = tok_emb + pos_emb
        #h_emb = h_text*h_num
        x = x*x_num

        x = self.decoder(x)
        logits = self.lm_head(x)
        num = self.num_head(x)
        return logits, num