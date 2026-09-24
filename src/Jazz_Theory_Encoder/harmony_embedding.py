import torch
import torch.nn as nn



class HarmonyEmbedding(nn.Module):

    def __init__(
        self,
        vocab_sizes,
        d_model=512
    ):
        super().__init__()
        # Discrete token embedding
        self.chord_embedding=nn.Embedding(vocab_sizes["chord"],64)
        self.root_embedding = nn.Embedding(12,16)
        self.bass_embedding = nn.Embedding(12,16)
        self.bass_interval_embedding = nn.Embedding(12,8)
        self.inversion_embedding = nn.Embedding(7,8)
        self.attribute_embedding = nn.Embedding(21,16)
        self.function_embedding = nn.Embedding(3,8)
        self.level_embedding = nn.Embedding(12,8)
        # self.cadence_embedding = nn.Embedding(9,8)
        self.scale_embedding = nn.Embedding(21,16)
        # self.style_embedding = nn.Embedding(11,8)
        # self.rhythm_embedding = nn.Embedding(9,8)
        # self.phrase_embedding = nn.Embedding(6,8)
        self.duration_embedding=nn.Embedding(vocab_sizes["duration"],8)
        self.beat_embedding=nn.Embedding(vocab_sizes["beat"],8)
        self.section_embedding=nn.Embedding(vocab_sizes["section"],16)
        self.time_embedding=nn.Embedding(vocab_sizes["time"],4)

        # Music theory features
        self.scale_encoder = nn.Linear(12,32)
        self.chord_tone_encoder = nn.Linear(12,32)
        self.guide_encoder = nn.Linear(12,32)
        self.tension_encoder = nn.Linear(12,32)
        self.available_tension_encoder = nn.Linear(12,32)
        self.avoid_encoder = nn.Linear(12,32)
        self.fusion=nn.Linear(388,d_model)

    def forward(self,input_ids,scale_vector,chord_tones_vector,guide_vector,tension_vector,available_tension_vector,avoid_vector):
        """
        x:

        batch,seq,5

        """
        chord = input_ids[:,:,0]
        root = input_ids[:,:,1]
        bass = input_ids[:,:,2]
        bass_interval = input_ids[:,:,3]
        inversion = input_ids[:,:,4]
        attribute = input_ids[:,:,5]
        function = input_ids[:,:,6]
        level = input_ids[:,:,7]
        # cadence = input_ids[:,:,8]
        scale = input_ids[:,:,8]
        # style = input_ids[:,:,10]
        # rhythm = input_ids[:,:,11]
        # phrase = input_ids[:,:,12]
        duration = input_ids[:,:,9]
        beat = input_ids[:,:,10]
        section = input_ids[:,:,11]
        time = input_ids[:,:,12]

        embeddings=[
            self.chord_embedding(chord),
            self.root_embedding(root),
            self.bass_embedding(bass),
            self.bass_interval_embedding(bass_interval),
            self.inversion_embedding(inversion),
            self.attribute_embedding(attribute),
            self.function_embedding(function),
            self.level_embedding(level),
            # self.cadence_embedding(cadence),
            self.scale_embedding(scale),
            # self.style_embedding(style),
            # self.rhythm_embedding(rhythm),
            # self.phrase_embedding(phrase),
            self.duration_embedding(duration),
            self.beat_embedding(beat),
            self.section_embedding(section),
            self.time_embedding(time)
        ]
        x = torch.cat(embeddings,dim=-1)
        theory=torch.cat(
            [
            self.scale_encoder(scale_vector),
            self.chord_tone_encoder(chord_tones_vector),
            self.guide_encoder(guide_vector),
            self.tension_encoder(tension_vector),
            self.available_tension_encoder(available_tension_vector),
            self.avoid_encoder(avoid_vector)
            ],
            dim=-1
        )
        x = torch.cat([x,theory],dim=-1)
        x = self.fusion(x)
        return x
