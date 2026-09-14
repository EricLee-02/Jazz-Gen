import torch
import json


from melody_generate_model import JazzGenerationModel
from Jazz_Theory_Encoder.harmony_model import HarmonyModel
from Jazz_Melody_Decoder.melody_decoder_Transformer import JazzTransformer

from midi_decoder import token_to_midi



DEVICE=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



CHECKPOINT="/content/drive/MyDrive/JazzGen_Data/check_point/generation/generation_best.pt"

VOCAB_FILE="/content/drive/MyDrive/JazzGen_Data/Melody_Vocabulary/vocabulary.json"



# ======================
# build model
# ======================


harmony_encoder=HarmonyModel(
    chord_vocab_size=1064,
    duration_vocab_size=10,
    beat_vocab_size=20,
    section_vocab_size=20,
    time_vocab_size=10,
    d_model=512,
    n_heads=8,
    num_layers=6
)



with open(VOCAB_FILE) as f:
    vocab=json.load(f)


vocab_size=len(
    vocab["token_to_id"]
)


melody_decoder=JazzTransformer(
    vocab_size=vocab_size,
    max_seq_len=512,
    d_model=512,
    n_heads=8,
    num_layers=8,
    dropout=0.1
)



model=JazzGenerationModel(
    harmony_encoder,
    melody_decoder
)



checkpoint=torch.load(
    CHECKPOINT,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.to(DEVICE)

print("Model loaded")



# ======================
# harmony input
# ======================

harmony={

"input_ids":...,

"scale_vector":...,

"chord_tones_vector":...,

"guide_vector":...,

"tension_vector":...,

"available_tension_vector":...,

"avoid_vector":...,

"attention_mask":...

}



# ======================
# generate
# ======================


BOS=vocab["token_to_id"]["<BOS>"]

EOS=vocab["token_to_id"]["<EOS>"]



tokens=model.generate(
    harmony,
    BOS,
    EOS,
    max_length=512,
    temperature=0.8,
    top_k=20
)


tokens=tokens[0].cpu().tolist()



print(tokens[:50])



# ======================
# MIDI
# ======================


token_to_midi(
    tokens,
    vocab,
    output="generated_jazz.mid"
)


print(
"Saved generated_jazz.mid"
)