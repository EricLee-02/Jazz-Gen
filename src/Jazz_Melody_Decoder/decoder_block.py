import torch.nn as nn
from jazz_attention import JazzAttention, CrossAttention



class DecoderBlock(nn.Module):
    def __init__(self,d_model,heads,ff_dim,dropout=0.1):
        super().__init__()
        self.attn=JazzAttention(d_model,heads,dropout=dropout)
        self.cross_atten = CrossAttention(d_model,heads,dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model) 
        self.ff=nn.Sequential(
            nn.Linear(d_model,ff_dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim*4,d_model)
            )
        self.dropout=nn.Dropout(dropout)


    def forward(self,x,memory,mask=None):
        x=x+self.dropout(self.attn(self.norm1(x),jazz_mask = mask))
        x = x+self.dropout(self.cross_atten(self.norm2(x),memory))
        x=x+self.dropout(self.ff(self.norm3(x)))
        return x