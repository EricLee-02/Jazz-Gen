"""Generate solo tokens/MIDI using a single constrained generation entry point.

Run in the original package, e.g. python -m src.generate_jazz.
Replace melody_generate_model.py in the same directory along with this file.
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
    HARMONY_CHECKPOINT_FILE,
    MELODY_CHECKPOINT_FILE,
    GENERATION_CHECKPOINT_FILE,
    MELODY_VOCAB_FILE,
    OUTPUT_DIR,
)


# Same model dimensions and sampling parameters as the supplied script.
MAX_LENGTH = 2048
TEMPERATURE = 0.75             # Set 0 for an argmax diagnostic comparison.
TOP_K = 20
SEED = 42                      # Same environment/settings -> reproducible sampling.
BEATS_PER_BAR = None            # None: use vocabulary; set 4 for a confirmed 4/4 grid.
START_BAR = None                # Preserve model's choice; set 1 to start at BAR_1.
MAX_BAR = None                  # Optional upper bound in YOUR melody bar numbering.


def load_checkpoint(module, path, label):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"{label}: expected a state_dict or model_state_dict checkpoint.")
    if not set(dict(module.named_parameters())).intersection(state):
        raise RuntimeError(f"{label}: no parameter names match this model. Check the file/prefixes.")
    # Keep the original partial-load behavior, but make mismatches visible.
    result = module.load_state_dict(state, strict=False)
    print(f"{label} checkpoint loaded: {path}")
    if result.missing_keys:
        print(f"  WARNING missing keys ({len(result.missing_keys)}): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  WARNING unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(MELODY_VOCAB_FILE, "r", encoding="utf8") as f:
        vocab = json.load(f)
    token_to_id = vocab["token_to_id"]
    id_to_token = {int(k): v for k, v in vocab["id_to_token"].items()}
    if (len(id_to_token) != len(token_to_id)
            or set(token_to_id.values()) != set(range(len(token_to_id)))
            or any(id_to_token.get(idx) != token for token, idx in token_to_id.items())):
        raise ValueError("Vocabulary IDs must be contiguous and both mappings must agree.")
    bos_id = token_to_id["<BOS>"]
    eos_id = token_to_id["<EOS>"]
    print("Vocabulary size:", len(token_to_id))
    print("Loading models...")

    harmony_encoder = HarmonyModel(
        chord_vocab_size=1064, duration_vocab_size=10, beat_vocab_size=20,
        section_vocab_size=20, time_vocab_size=10, d_model=512,
        num_heads=8, num_layers=6,
    )
    melody_decoder = JazzTransformer(
        vocab_size=len(token_to_id), max_seq_len=2048,
        d_model=512, num_heads=8, num_layers=8, dropout=0.1,
    )
    model = JazzGenerationModel(harmony_encoder, melody_decoder)
    load_checkpoint(harmony_encoder, HARMONY_CHECKPOINT_FILE,)
    load_checkpoint(melody_decoder, MELODY_CHECKPOINT_FILE)
    # Matching encoder/decoder keys in this final checkpoint override the above.
    load_checkpoint(model, GENERATION_CHECKPOINT_FILE)
    model = model.to(device)
    model.eval()

    chords = [
        "CMaj7", "CMin7 F7", "BbMaj7", "BbMin7 Eb7",
        "AbMaj7", "DMin7 G7#9", "CMaj7", "CMaj7",
        "DMin7", "G7", "CMaj7/E", "A7",
        "DMin7", "G7", "CMaj7", "DMin7 G7",
        "CMaj7", "CMin7 F7", "BbMaj7", "BbMin7 Eb7",
        "AbMaj7", "DMin7 G7#9", "CMaj7", "CMaj7",
    ]
    harmony = HarmonyGenerator().build(chords)
    harmony = {k: value.to(device) for k, value in harmony.items()}
    print("Harmony prepared")
    print("Generating...")

    # Only one generation loop: the method in melody_generate_model.py.
    generated, info = model.generate(
        harmony, bos_id=bos_id, eos_id=eos_id, max_length=MAX_LENGTH,
        temperature=TEMPERATURE, top_k=TOP_K, token_to_id=token_to_id,
        beats_per_bar=BEATS_PER_BAR, start_bar=START_BAR, max_bar=MAX_BAR,
        seed=SEED, return_info=True,
    )
    tokens = [id_to_token[i] for i in generated[0].cpu().tolist()]
    # No token-by-token deletion after generation: preserve event boundaries.
    audit = audit_melody_tokens(tokens)
    if (any(value for key, value in audit.items() if key != "note_count")
            or audit["note_count"] != info["note_count"]):
        raise RuntimeError(f"Time validation failed before MIDI export: {audit}")
    info["audit"] = audit

    token_output = os.path.join(OUTPUT_DIR, "generated_tokens.json")
    report_output = os.path.join(OUTPUT_DIR, "generation_report.json")
    midi_output = os.path.join(OUTPUT_DIR, "generated_jazz.mid")
    with open(token_output, "w", encoding="utf8") as f:
        json.dump(tokens, f, indent=2, ensure_ascii=False)
    with open(report_output, "w", encoding="utf8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print("Saved tokens:", token_output)
    print("Generation report:", json.dumps(info, ensure_ascii=False))
    if info["stop_reason"] == "token_budget":
        print("Reached token budget; output ends at a complete note, not necessarily the final harmony bar.")
    midi = tokens_to_midi(tokens)
    midi.save(midi_output)
    print("Saved MIDI:", midi_output)


if __name__ == "__main__":
    main()
