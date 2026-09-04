import torch
import torch.nn as nn

from decoder_block import DecoderBlock



class JazzTransformer(nn.Module):
    def __init__(self,vocab_size,max_seq_len=512,d_model=512,num_layers=8,n_heads=8,dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size,d_model)
        # position embedding
        self.position_embedding = nn.Embedding(max_seq_len,d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([DecoderBlock(d_model,n_heads,d_model*4,dropout)for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model,vocab_size)
        self.max_seq_len=max_seq_len


    def forward(self,tokens):

        print("Transformer input shape:",tokens.shape)
        print("max token:",tokens.max().item(),"min token:",
        tokens.min().item())
        print("embedding size:",self.token_embedding.num_embeddings)

        B,T=tokens.shape

        assert T <= self.max_seq_len, (f"Sequence length {T} > max {self.max_seq_len}")
        assert tokens.max().item() < self.token_embedding.num_embeddings, (f"Token id {tokens.max().item()} exceeds vocab "f"{self.token_embedding.num_embeddings}")

        assert tokens.min().item() >= 0, (f"Negative token id {tokens.min().item()}")
        x=self.token_embedding(tokens)

        # token embedding
   
        # position
        positions=torch.arange(T,device=tokens.device)
        pos=self.position_embedding(positions)
        x=x+pos
        x=self.dropout(x)
        for block in self.blocks:
            x=block(x)
        x=self.norm(x)
        logits=self.output(x)
        return logits