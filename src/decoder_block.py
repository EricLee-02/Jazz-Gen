import torch.nn as nn
from jazz_attention import JazzAttention



class DecoderBlock(nn.Module):
    def __init__(self,d_model,heads,ff_dim,dropout=0.1):
        super().__init__()
        self.attn=JazzAttention(d_model,heads)
        self.norm1=nn.LayerNorm(d_model)
        self.ff=nn.Sequential(
            nn.Linear(d_model,ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim,d_model)
            )
        self.norm2=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)


    def forward(self,x,mask=None):
        x=x+self.dropout(self.attn(self.norm1(x),mask))
        x=x+self.dropout(self.ff(self.norm2(x)))
        return x