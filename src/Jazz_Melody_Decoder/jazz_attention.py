import torch
import torch.nn as nn
import math
from src.config import MELODY_MAX_SEQ_LEN

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

        # self.register_buffer("causal_mask",torch.triu(torch.ones(2048,2048),diagonal=1).bool())



    def forward(self,x,jazz_mask=None):

        B,T,C=x.shape

        assert C == self.d_model, (f"Attention dim error {C}")

        assert T <= MELODY_MAX_SEQ_LEN, (f"Sequence too long {T}")


        Q=self.q_proj(x)
        K=self.k_proj(x)
        V=self.v_proj(x)

        Q=Q.view(B,T,self.heads,self.head_dim).transpose(1,2)
        K=K.view(B,T,self.heads,self.head_dim).transpose(1,2)
        V=V.view(B,T,self.heads,self.head_dim).transpose(1,2)

        score=torch.matmul(Q,K.transpose(-2,-1))

        assert score.shape[-1]==T
        assert score.shape[-2]==T

        score/=math.sqrt(self.head_dim)

        # causal mask
        causal=torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()

        score=score.masked_fill(causal,float("-inf"))

        # jazz theory mask
        if jazz_mask is not None:
            print("score:",score.shape,"jazz_mask:",jazz_mask.shape)

            score+=jazz_mask

        attn=torch.softmax(score,dim=-1)
        attn = self.attn_dropout(attn)

        out=torch.matmul(attn,V)
        out=out.transpose(1,2).contiguous()

        out=out.view(B,T,C)
        
        return self.out_proj(out)
    

class CrossAttention(nn.Module):

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
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,memory,memory_mask=None):
        """
        x:
        melody hidden
        [B,T,C]
        memory:
        harmony encoder output
        [B,S,C]
        """
        B,T,C=x.shape
        S=memory.shape[1]
        Q=self.q_proj(x)
        K=self.k_proj(memory)
        V=self.v_proj(memory)
        Q=Q.view(B,T,self.heads,self.head_dim).transpose(1,2)
        K=K.view(B,S,self.heads,self.head_dim).transpose(1,2)
        V=V.view(B,S,self.heads,self.head_dim).transpose(1,2)
        score=torch.matmul(Q,K.transpose(-2,-1))
        score/=math.sqrt(self.head_dim)
        if memory_mask is not None:
            score=score.masked_fill(memory_mask,float("-inf"))
        attn=torch.softmax(score,dim=-1)
        attn=self.dropout(attn)
        out=torch.matmul(attn,V)
        out=out.transpose(1,2).contiguous()
        out=out.view(B,T,C)
        return self.out_proj(out)