import torch
import torch.nn as nn
from .harmony_embedding import HarmonyEmbedding
from .harmony_transformer_encode import HarmonyEncoder
from .harmony_prediction import HarmonyPredictionHead



class HarmonyModel(nn.Module):
    def __init__(
        self,
        # vocab sizes
        chord_vocab_size,
        duration_vocab_size,
        beat_vocab_size,
        section_vocab_size,
        time_vocab_size,
        # embedding
        d_model=512,
        # transformer
        num_heads=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=512
    ):
        super().__init__()

        # ==========================
        # Harmony Embedding
        # ==========================

        self.embedding = HarmonyEmbedding(
            vocab_sizes={
            "chord":chord_vocab_size,
            "duration": duration_vocab_size,
            "beat": beat_vocab_size,
            "section":section_vocab_size,
            "time":time_vocab_size
            },
            d_model=d_model
        )



        # ==========================
        # Transformer Encoder
        # ==========================

        self.encoder = HarmonyEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len
        )



        # ==========================
        # Prediction Head
        # ==========================

        self.prediction = HarmonyPredictionHead(
            d_model=d_model,
            chord_vocab_size=chord_vocab_size
        )


    def encode(self,
               input_ids,
               scale_vector,
               chord_tones_vector,
               guide_vector,
               tension_vector,
               available_tension_vector,
               avoid_vector,
               attention_mask = None):
        x = self.embedding(
            input_ids,
            scale_vector,
            chord_tones_vector,
            guide_vector,
            tension_vector,
            available_tension_vector,
            avoid_vector,
        )
        x = self.encoder(x,attention_mask)
        return x




    def forward(
        self,
        input_ids,
        scale_vector,
        chord_tones_vector,
        guide_vector,
        tension_vector,
        available_tension_vector,
        avoid_vector,
        attention_mask=None
    ):
        # ==========================
        # Embedding
        # ==========================

        memory = self.encode(
            input_ids,
            scale_vector,
            chord_tones_vector,
            guide_vector,
            tension_vector,
            available_tension_vector,
            avoid_vector,
            attention_mask
        )



        # ==========================
        # Encoder
        # ==========================

        # x = self.encoder(x,attention_mask)
        # ==========================
        # Prediction
        # ==========================
        # logits = self.prediction(x)
        return memory