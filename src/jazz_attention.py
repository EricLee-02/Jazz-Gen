import torch
import torch.nn as nn
import math


class JazzAttention(nn.Module):
    def __init__(self,d_model,heads,dropout=0.1):
        super().__init__()
        
        assert d_model % heads == 0

        self.d_model=d_model
        self.heads=heads
        self.head_dim=d_model//heads
        self.q_proj=nn.Linear(d_model,d_model)
        self.k_proj=nn.Linear(d_model,d_model)
        self.v_proj=nn.Linear(d_model,d_model)
        self.out_proj=nn.Linear(d_model,d_model)
        self.attn_dropout=nn.Dropout(dropout)

        self.register_buffer("causal_mask",torch.triu(torch.ones(2048,2048),diagonal=1).bool())



    def forward(self,x,jazz_mask=None):

        B,T,C=x.shape
        Q=self.q_proj(x)
        K=self.k_proj(x)
        V=self.v_proj(x)

        Q=Q.view(B,T,self.heads,self.head_dim).transpose(1,2)
        K=K.view(B,T,self.heads,self.head_dim).transpose(1,2)
        V=V.view(B,T,self.heads,self.head_dim).transpose(1,2)

        score=torch.matmul(Q,K.transpose(-2,-1))
        score/=math.sqrt(self.head_dim)

        # causal mask
        causal=torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()

        score=score.masked_fill(causal,float("-inf"))

        # jazz theory mask
        if jazz_mask is not None:
            score+=jazz_mask

        attn=torch.softmax(score,dim=-1)
        attn = self.attn_dropout(attn)

        out=torch.matmul(attn,V)
        out=out.transpose(1,2).contiguous()

        out=out.view(B,T,C)
        
        return self.out_proj(out)