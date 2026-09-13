import torch
import torch.nn as nn
import math



# ==================================================
# Positional Encoding
# ==================================================

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=512):
        super().__init__()
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # [1,max_len,d_model]

        self.register_buffer("pe",pe)

    def forward(self,x):

        """
        x:
        [batch,seq,d_model]
        """
        x = x + self.pe[:,:x.size(1)]
        return x


# ==================================================
# Harmony Transformer Encoder
# ==================================================

class HarmonyEncoder(nn.Module):
    def __init__(self,d_model=512,n_heads=8,num_layers=6,dim_feedforward=2048,dropout=0.1, max_len=512):
        super().__init__()
        # position
        self.position_encoding = PositionalEncoding(d_model,max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x,attention_mask=None):

        """
        x:

        [batch,seq,512]


        attention_mask:

        True = valid

        False = padding

        """
        x = self.position_encoding(x)
        # Transformer要求:
        # True = ignore
        if attention_mask is not None:
            padding_mask = ~attention_mask

        else:
            padding_mask=None
        x = self.encoder( x,src_key_padding_mask=padding_mask)
        x = self.norm(x)
        return x