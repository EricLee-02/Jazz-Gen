import os
import json
import torch


from .melody_generate_model import JazzGenerationModel
from .Jazz_Theory_Encoder.harmony_model import HarmonyModel
from .Jazz_Melody_Decoder.melody_decoder_Transformer import JazzTransformer
from .harmony_generator import HarmonyGenerator
from .midi_decoder import tokens_to_midi


from .config import (
    HARMONY_CHECKPOINT_FILE,
    MELODY_CHECKPOINT_FILE,
    GENERATION_CHECKPOINT_FILE,
    MELODY_VOCAB_FILE,
    OUTPUT_DIR
)



# ==================================================
# Device
# ==================================================

DEVICE=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



TOKEN_OUTPUT=os.path.join(
    OUTPUT_DIR,
    "generated_tokens.json"
)


MIDI_OUTPUT=os.path.join(
    OUTPUT_DIR,
    "generated_jazz.mid"
)



# ==================================================
# Vocabulary
# ==================================================

with open(
    MELODY_VOCAB_FILE,
    "r",
    encoding="utf8"
) as f:

    vocab=json.load(f)



token_to_id=vocab["token_to_id"]


id_to_token={
    int(k):v
    for k,v in vocab["id_to_token"].items()
}



BOS_ID=token_to_id["<BOS>"]
EOS_ID=token_to_id["<EOS>"]



print(
    "Vocabulary size:",
    len(token_to_id)
)




# ==================================================
# Build Models
# ==================================================

print("Loading models...")



# ----------------------
# Harmony Encoder
# ----------------------

harmony_encoder=HarmonyModel(
    chord_vocab_size=1064,
    duration_vocab_size=10,
    beat_vocab_size=20,
    section_vocab_size=20,
    time_vocab_size=10,
    d_model=512,
    num_heads=8,
    num_layers=6
)



# ----------------------
# Melody Decoder
# ----------------------

melody_decoder=JazzTransformer(
    vocab_size=len(token_to_id),
    max_seq_len=2048,
    d_model=512,
    num_heads=8,
    num_layers=8,
    dropout=0.1
)



model=JazzGenerationModel(
    harmony_encoder,
    melody_decoder
)



# ==================================================
# Load checkpoints
# ==================================================


# ---------- Harmony ----------

checkpoint=torch.load(
    HARMONY_CHECKPOINT_FILE,
    map_location=DEVICE
)


if "model_state_dict" in checkpoint:
    state=checkpoint["model_state_dict"]
else:
    state=checkpoint



harmony_encoder.load_state_dict(
    state,
    strict=False
)



print("Harmony checkpoint loaded")



# ---------- Melody ----------


checkpoint=torch.load(
    MELODY_CHECKPOINT_FILE,
    map_location=DEVICE
)


if "model_state_dict" in checkpoint:
    state=checkpoint["model_state_dict"]
else:
    state=checkpoint



melody_decoder.load_state_dict(
    state,
    strict=False
)



print("Melody checkpoint loaded")



# ---------- Generation ----------


checkpoint=torch.load(
    GENERATION_CHECKPOINT_FILE,
    map_location=DEVICE
)



if "model_state_dict" in checkpoint:

    state=checkpoint["model_state_dict"]

else:

    state=checkpoint



model.load_state_dict(
    state,
    strict=False
)



print("Generation checkpoint loaded")



harmony_encoder=harmony_encoder.to(DEVICE)
melody_decoder=melody_decoder.to(DEVICE)
model=model.to(DEVICE)



harmony_encoder.eval()
melody_decoder.eval()
model.eval()



# ==================================================
# Harmony condition
# ==================================================

chords=[
    "CMaj7",
    "CMin7 F7",
    "BbMaj7"
]



harmony_builder=HarmonyGenerator()


harmony=harmony_builder.build(
    chords
)


for k in harmony:

    harmony[k]=harmony[k].to(DEVICE)



print("Harmony prepared")




# ==================================================
# Token Constraint
# ==================================================


def clean_generated_tokens(tokens):

    """
    Melody constraint:

    1. Remove harmony tokens
    2. One pitch per position
    3. Prevent bar backward
    """


    cleaned=[]


    current_position=None
    note_generated=False

    last_bar=-1



    remove_prefix=[
        "CHORD",
        "ROOT",
        "ROMAN",
        "FUNCTION",
        "QUALITY",
        "SCALE"
    ]



    for token in tokens:


        # remove harmony output

        if any(
            token.startswith(x)
            for x in remove_prefix
        ):

            continue



        # BAR check

        if token.startswith("BAR"):


            try:

                bar=int(
                    token.split("_")[1]
                )


                if bar < last_bar:

                    continue


                last_bar=bar


            except:

                pass



        # POSITION

        if token.startswith(
            "POSITION"
        ):

            current_position=token
            note_generated=False



        # pitch constraint

        if token.startswith(
            "PITCH"
        ):


            if note_generated:

                continue


            note_generated=True



        cleaned.append(token)



    return cleaned





# ==================================================
# Generation
# ==================================================


@torch.no_grad()
def generate(
        model,
        harmony,
        max_length=2048,
        temperature=0.75,
        top_k=20
):


    memory=model.harmony_encoder.encode(
        **harmony
    )



    generated=torch.tensor(
        [[BOS_ID]],
        device=DEVICE
    )



    for step in range(
        max_length-1
    ):


        logits=model.melody_decoder(
            generated,
            memory
        )



        next_logits=logits[:,-1,:]


        next_logits/=temperature



        if top_k:


            values,indices=torch.topk(
                next_logits,
                top_k
            )


            probs=torch.softmax(
                values,
                dim=-1
            )


            sample=torch.multinomial(
                probs,
                1
            )


            next_token=indices.gather(
                1,
                sample
            )


        else:


            probs=torch.softmax(
                next_logits,
                dim=-1
            )


            next_token=torch.multinomial(
                probs,
                1
            )



        generated=torch.cat(
            [
                generated,
                next_token
            ],
            dim=1
        )



        if next_token.item()==EOS_ID:

            break



    return generated.squeeze(0).cpu().tolist()




# ==================================================
# Generate
# ==================================================


print("Generating...")


generated_ids=generate(
    model,
    harmony,
    max_length=2048,
    temperature=0.75,
    top_k=20
)



tokens=[
    id_to_token[i]
    for i in generated_ids
]



print(
    "Before cleaning:",
    len(tokens)
)



tokens=clean_generated_tokens(
    tokens
)



print(
    "After cleaning:",
    len(tokens)
)




# ==================================================
# Save tokens
# ==================================================

with open(
    TOKEN_OUTPUT,
    "w",
    encoding="utf8"
) as f:


    json.dump(
        tokens,
        f,
        indent=2,
        ensure_ascii=False
    )



print(
    "Saved tokens:",
    TOKEN_OUTPUT
)




# ==================================================
# MIDI
# ==================================================

midi=tokens_to_midi(
    tokens
)


midi.save(
    MIDI_OUTPUT
)



print(
    "Saved MIDI:",
    MIDI_OUTPUT
)