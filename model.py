import torch
from torch import nn




class Numformer(nn.Module):
    def __init__(
            self,
            vocab_size, #same hyperparameters as paper code
            d_model=160,
            nhead=5,
            num_layers=6,
            dim_feedforward=640,
            dropout=0.1,
            activation=nn.GELU(),
            layer_norm_eps=1e-05,
            batch_first=True,
            norm_first=True,
            context_length=224):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first)
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False)
        
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


    def forward(self, x, x_num):
        batch_dim, time_dim = x.shape

        tok_emb = self.token_embed_table(x)
        #pos_emb takes as input a tensor with values from 0 to time_dim-1 (time_dim == context_length)
        pos_emb = self.position_embed_table(torch.arange(time_dim).to(tok_emb.device))
        
        x = tok_emb + pos_emb
        #h_emb = h_text*h_num
        x = x*x_num.unsqueeze(-1)

        x = self.encoder(x)
        logits = self.lm_head(x)
        num = self.num_head(x)
        return logits, num