import torch
import torch.nn as nn



class JazzGenerationModel(nn.Module):

    def __init__(self,harmony_encoder,melody_decoder):
        super().__init__()
        # Harmony Theory Encoder
        self.harmony_encoder = harmony_encoder
        for p in self.harmony_encoder.parameters():
            p.requires_grad=False
        # Melody Transformer Decoder
        self.melody_decoder = melody_decoder


    def forward(self,harmony,melody_input):

        """
        harmony:

        {
            input_ids,
            scale_vector,
            chord_tones_vector,
            guide_vector,
            tension_vector,
            available_tension_vector,
            avoid_vector,
            attention_mask
        }


        melody_input:

        [B,T]

        """


        # ==========================
        # Harmony Encoding
        # ==========================

        memory = self.harmony_encoder.encode( **harmony)


        """
        memory:

        [B,H,512]

        """



        # ==========================
        # Melody Generation
        # ==========================
        logits = self.melody_decoder(melody_input,memory)


        """
        logits:

        [B,T,vocab_size]

        """


        return logits

    @torch.no_grad()
    def encode_harmony(self,harmony):

        """
        inference时单独获取Harmony memory
        """
        memory=self.harmony_encoder.encode( **harmony)

        return memory