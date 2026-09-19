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
    OUTPUT_DIR,
    TOKENS_PER_BAR,
    GENERATE_32_BAR_LENGTH,
    GENERATE_48_BAR_LENGTH,
    GENERATE_64_BAR_LENGTH,
)


# ===============================
# Generation Config
# ===============================

TEMPERATURE = 0.85
TOP_K = 40

SEED = 42

BEATS_PER_BAR = None

START_BAR = None


# ===============================
# Load checkpoint
# ===============================

def load_checkpoint(module, path, label="Model"):

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]

    else:
        state = checkpoint


    result = module.load_state_dict(
        state,
        strict=False
    )


    print(f"{label} loaded:")
    print(path)


    if result.missing_keys:
        print(
            "Missing keys:",
            len(result.missing_keys)
        )

    if result.unexpected_keys:
        print(
            "Unexpected keys:",
            len(result.unexpected_keys)
        )



# ===============================
# Prompt
# ===============================

def build_prompt(token_to_id, key, tempo):

    key_token = f"KEY_{key}"


    if key_token not in token_to_id:
        raise ValueError(
            f"Missing key token: {key_token}"
        )


    tempo_candidates = [
        f"AVGTEMPO_{tempo}",
        f"AVGTEMPO_{float(tempo):.1f}"
    ]


    tempo_token = None

    for t in tempo_candidates:

        if t in token_to_id:
            tempo_token = t
            break


    if tempo_token is None:

        raise ValueError(
            f"Tempo token not found: {tempo}"
        )


    tokens = [
        "<BOS>",
        key_token,
        tempo_token
    ]


    ids = [
        token_to_id[t]
        for t in tokens
    ]


    return torch.tensor(
        [ids],
        dtype=torch.long
    )



# ===============================
# Length control
# ===============================

def get_max_length(bars):

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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    # ---------------------------
    # Vocabulary
    # ---------------------------

    with open(
        MELODY_VOCAB_FILE,
        "r",
        encoding="utf8"
    ) as f:

        vocab = json.load(f)



    token_to_id = vocab["token_to_id"]

    id_to_token = {
        int(k):v
        for k,v in vocab["id_to_token"].items()
    }


    bos_id = token_to_id["<BOS>"]

    eos_id = token_to_id["<EOS>"]


    print(
        "Vocabulary size:",
        len(token_to_id)
    )


    # ---------------------------
    # Build model
    # ---------------------------


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

        max_seq_len=2048,

        d_model=512,

        num_heads=8,

        num_layers=8,

        dropout=0.1
    )



    model = JazzGenerationModel(
        harmony_encoder,
        melody_decoder
    )


    # IMPORTANT:
    # Load end-to-end trained checkpoint

    load_checkpoint(
        model,
        GENERATION_CHECKPOINT_FILE,
        "Generation Model"
    )


    model.to(device)

    model.eval()



    # ===========================
    # User Condition
    # ===========================


    bars = 32

    key = "Bb-maj"

    tempo = 120.1



    # ---------------------------
    # Chord progression
    # one chord per position
    # ---------------------------


    chords = [

        "CMin7",
        "F7",
        "BbMaj7",
        "EbMaj7",

        "AMin7b5",
        "D7",
        "GMin7",
        "C7",

        "FMin7",
        "Bb7",
        "EbMaj7",
        "A7",

        "DMin7",
        "G7",
        "CMaj7",
        "CMaj7",

    ]


    # repeat to 32 bars

    chords = chords * 2



    prompt = build_prompt(
        token_to_id,
        key,
        tempo
    )


    harmony = HarmonyGenerator().build(
        chords
    )


    harmony = {
        k:v.to(device)
        for k,v in harmony.items()
    }



    print("Harmony prepared")

    print("Generating...")



    # ===========================
    # Generate
    # ===========================


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

        max_bar=bars-1,

        seed=SEED,

        return_info=True
    )



    tokens = [

        id_to_token[i]

        for i in generated[0].cpu().tolist()

    ]



    # ===========================
    # Audit
    # ===========================


    audit = audit_melody_tokens(tokens)


    if (
        any(
            value
            for key,value in audit.items()
            if key!="note_count"
        )

        or audit["note_count"]
        != info["note_count"]
    ):

        raise RuntimeError(
            f"Time validation failed:{audit}"
        )


    info["audit"] = audit



    # ===========================
    # Save
    # ===========================


    token_output = os.path.join(
        OUTPUT_DIR,
        "generated_tokens.json"
    )


    report_output = os.path.join(
        OUTPUT_DIR,
        "generation_report.json"
    )


    midi_output = os.path.join(
        OUTPUT_DIR,
        "generated_jazz.mid"
    )



    with open(
        token_output,
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            tokens,
            f,
            indent=2,
            ensure_ascii=False
        )



    with open(
        report_output,
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            info,
            f,
            indent=2,
            ensure_ascii=False
        )



    print(
        "Saved tokens:",
        token_output
    )


    print(
        json.dumps(
            info,
            ensure_ascii=False
        )
    )


    midi = tokens_to_midi(tokens)

    midi.save(
        midi_output
    )


    print(
        "Saved MIDI:",
        midi_output
    )



if __name__=="__main__":

    main()