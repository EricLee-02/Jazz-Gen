import torch
import torch.nn as nn


class HarmonyPredictionHead(nn.Module):

    def __init__(self,d_model=512,chord_vocab_size=1063):
        super().__init__()
        # 预测前LayerNorm
        self.norm = nn.LayerNorm(d_model)

        # hidden representation -> chord probability
        self.output = nn.Linear(d_model,chord_vocab_size)

    def forward(self, x):

        """
        x:
        Transformer Encoder output

        Shape:
        [batch, seq_len, d_model]

        Example:
        [8,128,512]
        """
        x = self.norm(x)
        logits = self.output(x)

        """
        logits:

        [batch, seq_len, chord_vocab_size]

        Example:
        [8,128,1063]
        """
        return logits