# """Generate solo tokens/MIDI using end-to-end generation model.

# Run:
# python -m src.generate_jazz
# """


# import json
# import os
# import torch

# from .melody_generate_model import JazzGenerationModel, audit_melody_tokens
# from .Jazz_Theory_Encoder.harmony_model import HarmonyModel
# from .Jazz_Melody_Decoder.melody_decoder_Transformer import JazzTransformer
# from .harmony_generator import HarmonyGenerator
# from .midi_decoder import tokens_to_midi


# from .config import (
#     GENERATION_CHECKPOINT_FILE,
#     MELODY_VOCAB_FILE,
#     MELODY_CHECKPOINT_FILE,
#     HARMONY_CHECKPOINT_FILE,
#     OUTPUT_DIR,
#     TOKENS_PER_BAR,

#     MELODY_MAX_SEQ_LEN,
#     MELODY_D_MODEL,
#     MELODY_HEADS,
#     MELODY_LAYERS,
#     DROPOUT,

#     GENERATE_32_BAR_LENGTH,
#     GENERATE_48_BAR_LENGTH,
#     GENERATE_64_BAR_LENGTH,
# )


# # ===============================
# # Generation Config
# # ===============================

# TEMPERATURE = 0.95
# TOP_K = 50
# SEED = None
# BEATS_PER_BAR = 4
# START_BAR = 0
# SWING = True
# SWING_RATIO = 0.625



# # ===============================
# # Prompt
# # ===============================
# def build_prompt(token_to_id,key,tempo):
#     key_token = f"KEY_{key}"

#     if key_token not in token_to_id:
#         raise ValueError(f"Missing key token: {key_token}")
    
#     tempo_tokens = [token for token in token_to_id if token.startswith("AVGTEMPO_")]
#     if not tempo_tokens:
#         raise ValueError("No AVGTEMPO token found in vocabulary")
#     tempo_token = min(tempo_tokens,key=lambda token:abs(float(token.split("_",1)[1])-float(tempo)))
#     print(f"Tempo condition: {tempo} -> {tempo_token}")
#     tokens = ["<BOS>", key_token, tempo_token]
#     ids = [token_to_id[token] for token in tokens]
#     return torch.tensor([ids], dtype= torch.long)


# # ===============================
# # Length control
# # ===============================

# def get_max_length(bars):
#     if bars == 16:
#         return 64*16
#     if bars == 32:
#         return GENERATE_32_BAR_LENGTH
#     if bars == 48:
#         return GENERATE_48_BAR_LENGTH
#     if bars == 64:
#         return GENERATE_64_BAR_LENGTH
#     return bars * TOKENS_PER_BAR


# # ===============================
# # Main
# # ===============================


# def main():
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs( OUTPUT_DIR, exist_ok=True)
#     # ===============================
#     # Vocabulary
#     # ===============================
#     with open( MELODY_VOCAB_FILE, "r",encoding="utf8") as f:
#         vocab=json.load(f)
#     token_to_id=vocab["token_to_id"]
#     id_to_token={int(k):v for k,v in vocab["id_to_token"].items() }
#     bos_id=token_to_id["<BOS>"]
#     eos_id=token_to_id["<EOS>"]
#     print("Vocabulary size:", len(token_to_id))
#     # ===============================
#     # Build Model
#     # ===============================

#     harmony_encoder=HarmonyModel(
#         chord_vocab_size=1064,
#         duration_vocab_size=10,
#         beat_vocab_size=20,
#         section_vocab_size=20,
#         time_vocab_size=10,
#         d_model=512,
#         num_heads=8,
#         num_layers=6
#     )

#     melody_decoder=JazzTransformer(
#         vocab_size=len(token_to_id),
#         max_seq_len=2048,
#         d_model=MELODY_D_MODEL,
#         num_heads=MELODY_HEADS,
#         num_layers=MELODY_LAYERS,
#         dropout=DROPOUT
#     )





#     model=JazzGenerationModel(harmony_encoder,melody_decoder)



#     # ===============================
#     # Load End-to-End checkpoint
#     # ===============================

#     # checkpoint=torch.load( GENERATION_CHECKPOINT_FILE, map_location="cpu")
#     # harmony_result = harmony_encoder.load_state_dict( checkpoint["harmony_state_dict"], strict=True)

#     # melody_result = melody_decoder.load_state_dict(checkpoint["melody_state_dict"], strict=True)


#     # print("===============================")
#     # print("Generation model loaded")
#     # print( "Epoch:", checkpoint["epoch"] )
#     # print( "Val Loss:",checkpoint["val_loss"])
#     # print("===============================")
#     # model=model.to(device)
#     # model.eval()

#     harmony_checkpoint = torch.load(HARMONY_CHECKPOINT_FILE,map_location="cpu")
#     harmony_state = harmony_checkpoint["encoder"]
#     harmony_result = harmony_encoder.load_state_dict(harmony_state,strict=False)
#     print("=" * 50)
#     print("Harmony Encoder Loaded")
#     print("Missing:", harmony_result.missing_keys)
#     print("Unexpected:", harmony_result.unexpected_keys)


#     melody_checkpoint = torch.load(MELODY_CHECKPOINT_FILE,map_location="cpu")
#     if "model_state_dict" in melody_checkpoint:
#         melody_state = melody_checkpoint["model_state_dict"]
#     elif "decoder" in melody_checkpoint:
#         melody_state = melody_checkpoint["decoder"]
#     else:
#         melody_state =melody_checkpoint
#     melody_result = melody_decoder.load_state_dict(melody_state,strict=True)
#     print("=" * 50)
#     print("Harmony Encoder Loaded")
#     if isinstance(melody_checkpoint,dict):
#         if "epoch" in melody_checkpoint:
#             print("Epoch:", melody_checkpoint["epoch"])
#         if "loss" in melody_checkpoint:
#             print("Val loss:", melody_checkpoint["loss"])
#     print("=" * 50)
#     model = model.to(device)
#     model.eval()
        

    




#     # ===============================
#     # Condition
#     # ===============================

#     bars=32
#     key="C-maj"
#     tempo=120.1
#     chords=[
#         "CMaj7""CMin7 F7","BbMaj7","BbMin7 Eb7","AbMaj7","DMin7 G7","CMaj7","CMaj7",
#         "DMin7","G7","C/EMaj7","A7","DMin7", "G7", "CMaj7","DMin7 G7",
#         "CMaj7""CMin7 F7","BbMaj7","BbMin7 Eb7","AbMaj7","DMin7 G7","CMaj7","DMin7 G7",
#     ]

#     prompt=build_prompt(token_to_id,key,tempo)
#     harmony_generator=HarmonyGenerator(
#         max_harmony_len = 512,
#         device = device
#     )
#     harmony = harmony_generator.build(chords)
#     harmony={ k:v.to(device) for k,v in harmony.items()}



#     print("Harmony prepared")
#     print("Generating...")

#     # ===============================
#     # Generate
#     # ===============================



#     generated,info=model.generate(
#         harmony,
#         prompt_ids=prompt.to(device),
#         eos_id=eos_id,
#         max_length=get_max_length(bars),
#         temperature=TEMPERATURE,
#         top_k=TOP_K,
#         token_to_id=token_to_id,
#         beats_per_bar=BEATS_PER_BAR,

#         start_bar = 0,
#         max_bar=bars-1,
#         seed=SEED,
#         return_info=True,
#         chords=chords,
#         harmony_generator=harmony_generator,
#         dynamic_harmony=True
#     )

#     tokens=[id_to_token[i] for i in generated[0].cpu().tolist()]

#     audit = audit_melody_tokens(tokens)
#     info["audit"] = audit
#     print("=" * 50)
#     print("Generate Audit")
#     print(json.dumps(audit,indent=2,ensure_ascii=False))
#     print("=" * 50)


#     # ===============================
#     # Save
#     # ===============================



#     token_output=os.path.join( OUTPUT_DIR,"generated_tokens.json")

#     report_output=os.path.join(OUTPUT_DIR,"generation_report.json")

#     midi_output=os.path.join(OUTPUT_DIR, "generated_jazz.mid")

#     with open( token_output,"w", encoding="utf8" ) as f:
#         json.dump(tokens, f,indent=2,ensure_ascii=False)


#     with open(report_output,"w", encoding="utf8") as f:
#         json.dump( info,f,indent=2,ensure_ascii=False)
#     print("Generation report:")
#     print(json.dumps( info, ensure_ascii=False))

#     midi=tokens_to_midi(tokens,tempo_bpm=tempo,swing = SWING, swing_ratio = SWING_RATIO)
#     midi.save(midi_output)
#     print( "Saved MIDI:",midi_output)


# if __name__=="__main__":

#     main()


"""Generate solo tokens/MIDI using end-to-end generation model.

Run:
python -m src.generate_jazz
"""

import json
import os
import torch

from .melody_generate_model import JazzGenerationModel, audit_melody_tokens
from .Jazz_Theory_Encoder.harmony_model import HarmonyModel
from .Jazz_Melody_Decoder.melody_decoder_Transformer import JazzTransformer
from .harmony_generator import HarmonyGenerator
from .midi_decoder import tokens_to_midi

from .config import (
    GENERATION_CHECKPOINT_FILE,
    MELODY_VOCAB_FILE,
    MELODY_CHECKPOINT_FILE,
    HARMONY_CHECKPOINT_FILE,
    OUTPUT_DIR,
    TOKENS_PER_BAR,
    MELODY_MAX_SEQ_LEN,
    MELODY_D_MODEL,
    MELODY_HEADS,
    MELODY_LAYERS,
    DROPOUT,
    GENERATE_32_BAR_LENGTH,
    GENERATE_48_BAR_LENGTH,
    GENERATE_64_BAR_LENGTH,
)


# ===============================
# Generation Config
# ===============================

TEMPERATURE = 0.95
TOP_K = 50
SEED = None
BEATS_PER_BAR = 4
START_BAR = 0

# MIDI rendering only; this does not change symbolic model tokens.
SWING = True
SWING_RATIO = 0.625
SWING_RATE = 0.92


# ===============================
# Prompt
# ===============================

def build_prompt(token_to_id, key, tempo):
    key_token = f"KEY_{key}"

    if key_token not in token_to_id:
        raise ValueError(f"Missing key token: {key_token}")

    tempo_tokens = [
        token for token in token_to_id
        if token.startswith("AVGTEMPO_")
    ]
    if not tempo_tokens:
        raise ValueError("No AVGTEMPO token found in vocabulary")

    tempo_token = min(
        tempo_tokens,
        key=lambda token: abs(
            float(token.split("_", 1)[1]) - float(tempo)
        ),
    )

    print(f"Tempo condition: {tempo} -> {tempo_token}")

    tokens = ["<BOS>", key_token, tempo_token]
    ids = [token_to_id[token] for token in tokens]
    return torch.tensor([ids], dtype=torch.long)


# ===============================
# Length control
# ===============================

def get_max_length(bars):
    if bars == 16:
        return 64 * 16
    if bars == 32:
        return GENERATE_32_BAR_LENGTH
    if bars == 48:
        return GENERATE_48_BAR_LENGTH
    if bars == 64:
        return GENERATE_64_BAR_LENGTH
    return bars * TOKENS_PER_BAR


# ===============================
# Main
# ===============================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ===============================
    # Vocabulary
    # ===============================
    with open(MELODY_VOCAB_FILE, "r", encoding="utf8") as f:
        vocab = json.load(f)

    token_to_id = vocab["token_to_id"]
    id_to_token = {int(k): v for k, v in vocab["id_to_token"].items()}
    eos_id = token_to_id["<EOS>"]

    print("Vocabulary size:", len(token_to_id))

    # ===============================
    # Build Model
    # ===============================

    harmony_encoder = HarmonyModel(
        chord_vocab_size=1064,
        duration_vocab_size=10,
        beat_vocab_size=20,
        section_vocab_size=20,
        time_vocab_size=10,
        d_model=512,
        num_heads=8,
        num_layers=6,
    )

    melody_decoder = JazzTransformer(
        vocab_size=len(token_to_id),
        max_seq_len=MELODY_MAX_SEQ_LEN,
        d_model=MELODY_D_MODEL,
        num_heads=MELODY_HEADS,
        num_layers=MELODY_LAYERS,
        dropout=DROPOUT,
    )

    model = JazzGenerationModel(harmony_encoder, melody_decoder)

    # ===============================
    # Load checkpoints
    # ===============================

    # harmony_checkpoint = torch.load(
    #     HARMONY_CHECKPOINT_FILE,
    #     map_location="cpu",
    # )
    # harmony_state = harmony_checkpoint["encoder"]
    # harmony_result = harmony_encoder.load_state_dict(
    #     harmony_state,
    #     strict=False,
    # )

    # print("=" * 50)
    # print("Harmony Encoder Loaded")
    # print("Missing:", harmony_result.missing_keys)
    # print("Unexpected:", harmony_result.unexpected_keys)

    # melody_checkpoint = torch.load(
    #     MELODY_CHECKPOINT_FILE,
    #     map_location="cpu",
    # )

    # if "model_state_dict" in melody_checkpoint:
    #     melody_state = melody_checkpoint["model_state_dict"]
    # elif "decoder" in melody_checkpoint:
    #     melody_state = melody_checkpoint["decoder"]
    # else:
    #     melody_state = melody_checkpoint

    # melody_result = melody_decoder.load_state_dict(
    #     melody_state,
    #     strict=True,
    # )

    # print("=" * 50)
    # print("Melody Decoder Loaded")
    # print("Missing:", melody_result.missing_keys)
    # print("Unexpected:", melody_result.unexpected_keys)

    # if isinstance(melody_checkpoint, dict):
    #     if "epoch" in melody_checkpoint:
    #         print("Epoch:", melody_checkpoint["epoch"])
    #     if "loss" in melody_checkpoint:
    #         print("Val loss:", melody_checkpoint["loss"])

    # print("=" * 50)

    # model = model.to(device)
    # model.eval()

    #     ===============================
    # Load End-to-End checkpoint
    # ===============================

    checkpoint=torch.load( GENERATION_CHECKPOINT_FILE, map_location="cpu")
    harmony_result = harmony_encoder.load_state_dict( checkpoint["harmony_state_dict"], strict=True)

    melody_result = melody_decoder.load_state_dict(checkpoint["melody_state_dict"], strict=True)


    print("===============================")
    print("Generation model loaded")
    print( "Epoch:", checkpoint["epoch"] )
    print( "Val Loss:",checkpoint["val_loss"])
    print("===============================")
    model=model.to(device)
    model.eval()

    # ===============================
    # Condition
    # ===============================

    bars = 32
    key = "C-maj"
    tempo = 120.1

    # IMPORTANT:
    # one STRING == one BAR.
    # Multiple chord symbols inside the same string share that bar evenly.
    #
    # Examples:
    #   "CMaj7"       -> whole bar CMaj7
    #   "CMin7 F7"    -> first half CMin7, second half F7
    #   "Am7 D7 G7 C7"-> one chord per beat in 4/4
    #
    # This 16-bar form is repeated only to keep this example exactly 32 bars.
    # Replace the second 16 bars if your real form is different.
    chords = [
        "CMaj7","CMin7 F7", "BbMaj7","BbMin7 Eb7","AbMaj7","DMin7 G7", "CMaj7","DMin7 G7",
        "CMaj7","CMin7 F7", "BbMaj7","BbMin7 Eb7","AbMaj7","DMin7 G7", "CMaj7","CMaj7",
        "DMin7","G7","C/EMaj7", "A7","DMin7","G7","CMaj7","DMin7 G7",
        "CMaj7","CMin7 F7", "BbMaj7","BbMin7 Eb7","AbMaj7","DMin7 G7", "CMaj7","CMaj7"
    ]

    if len(chords) != bars:
        raise ValueError(
            f"Chord progression must contain exactly {bars} bar strings; "
            f"got {len(chords)}"
        )

    prompt = build_prompt(token_to_id, key, tempo)

    harmony_generator = HarmonyGenerator(
        max_harmony_len=512,
        device=device,
    )

    # Global harmony context.  HarmonyGenerator.build() now expands a bar
    # such as "DMin7 G7" into two separate harmony events for the encoder.
    harmony = harmony_generator.build(chords)
    harmony = {k: v.to(device) for k, v in harmony.items()}

    print("Harmony prepared")
    print("Bars:", len(chords))
    print("Harmony events:", len(harmony_generator.flatten_progression(chords)))
    print("Generating...")

    # ===============================
    # Generate
    # ===============================

    generated, info = model.generate(
        harmony,
        prompt_ids=prompt.to(device),
        eos_id=eos_id,
        max_length=get_max_length(bars),
        temperature=TEMPERATURE,
        top_k=TOP_K,
        token_to_id=token_to_id,
        beats_per_bar=BEATS_PER_BAR,
        start_bar=START_BAR,
        max_bar=bars - 1,
        seed=SEED,
        return_info=True,
        chords=chords,
        harmony_generator=harmony_generator,
        dynamic_harmony=True,
    )

    tokens = [
        id_to_token[i]
        for i in generated[0].cpu().tolist()
    ]

    audit = audit_melody_tokens(tokens)
    info["audit"] = audit

    print("=" * 50)
    print("Generate Audit")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print("=" * 50)

    # ===============================
    # Save
    # ===============================

    token_output = os.path.join(OUTPUT_DIR, "generated_tokens.json")
    report_output = os.path.join(OUTPUT_DIR, "generation_report.json")
    midi_output = os.path.join(OUTPUT_DIR, "generated_jazz.mid")

    with open(token_output, "w", encoding="utf8") as f:
        json.dump(tokens, f, indent=2, ensure_ascii=False)

    with open(report_output, "w", encoding="utf8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("Generation report:")
    print(json.dumps(info, ensure_ascii=False))

    midi = tokens_to_midi(
        tokens,
        tempo_bpm=tempo,
        swing=SWING,
        swing_ratio=SWING_RATIO,
        swing_rate=SWING_RATE,
    )
    midi.save(midi_output)
    print("Saved MIDI:", midi_output)


if __name__ == "__main__":
    main()
