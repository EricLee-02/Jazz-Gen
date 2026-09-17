import os
import json
import torch
from .melody_generate_model import JazzGenerationModel
from .Jazz_Theory_Encoder.harmony_model import HarmonyModel
from .Jazz_Melody_Decoder.melody_decoder_Transformer import JazzTransformer
from .harmony_generator import HarmonyGenerator
from .midi_decoder import tokens_to_midi
from .config import HARMONY_CHECKPOINT_DIR,MELODY_VOCAB_FILE,OUTPUT_DIR,MELODY_CHECKPOINT_DIR,GENERATION_CHECKPOINT_DIR


# ==================================================
# Config
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
# Load Vocabulary
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
    "Vocabulary:",
    len(token_to_id)
)



# ==================================================
# Build Model
# ==================================================

print("Loading model...")



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



checkpoint=torch.load( HARMONY_CHECKPOINT_DIR,map_location=DEVICE)

if "model_state_dict" in checkpoint:
    state=checkpoint["model_state_dict"]
elif "encoder" in checkpoint:
    state=checkpoint["encoder"]
else:
    state=checkpoint


melody_decoder.load_state_dict(
    state,
    strict=False
)

checkpoint=torch.load(
    MELODY_CHECKPOINT_DIR,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:

    state=checkpoint["model_state_dict"]

elif "encoder" in checkpoint:

    state=checkpoint["encoder"]


else:

    state=checkpoint


melody_decoder.load_state_dict(
    state,
    strict=False
)

checkpoint=torch.load(
    GENERATION_CHECKPOINT_DIR,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:

    state=checkpoint["model_state_dict"]

elif "encoder" in checkpoint:

    state=checkpoint["encoder"]


else:

    state=checkpoint


model.load_state_dict(
    state,
    strict=False
)


harmony_encoder = harmony_encoder.to(DEVICE)
harmony_encoder.eval()

melody_decoder = melody_decoder.to(DEVICE)
melody_decoder.eval()

model = model.to(DEVICE)
model.eval()


print("Harmony Encoder loaded")


# ==================================================
# Build Harmony Condition
# ==================================================

# 用户输入和弦

chords=[
  "CMaj7","CMin7 F7","BbMaj7","BMin7 Eb7",
  "AbMaj7","DMin7 G7#9","CMaj7","CMaj7",
  "DMin7","G7","CMaj7/E","A7",
  "DMin7","G7","CMaj7","DMin7 G7",
  "CMaj7","CMin7 F7","BbMaj7","BMin7 Eb7",
  "AbMaj7","DMin7 G7#9","CMaj7","CMaj7",
]



harmony_builder=HarmonyGenerator()


harmony=harmony_builder.build(chords)

for k in harmony:
    harmony[k]=harmony[k].to(DEVICE)



print("Harmony prepared")



# ==================================================
# Sampling Generation
# ==================================================

@torch.no_grad()
def generate(model,harmony, max_length=1536,temperature=0.9,
top_k=40):


    memory=model.harmony_encoder.encode(**harmony)
    generated=torch.tensor([[BOS_ID]],device=DEVICE)

    for step in range(max_length-1):

        logits=model.melody_decoder(
            generated,
            memory
        )


        next_logits=logits[:,-1,:]


        # temperature

        next_logits/=temperature



        # ======================
        # Top K sampling
        # ======================

        if top_k:
            values,indices=torch.topk(next_logits,top_k)
            probs=torch.softmax(values,dim=-1)
            sample=torch.multinomial(probs,1)
            next_token=indices.gather(1,sample)


        else:
            probs=torch.softmax(next_logits, dim=-1)
            next_token=torch.multinomial(probs,1)

        generated=torch.cat([generated,next_token],dim=1)



        if next_token.item()==EOS_ID:

            break



    return generated.squeeze(0).cpu().tolist()



# ==================================================
# Generate
# ==================================================

print("Generating...")



generated_ids=generate(model,harmony,max_length=1536,
temperature=0.9,top_k=40)



tokens=[id_to_token[i]for i in generated_ids]


print("Generated tokens:",len(tokens))


# print(tokens[:80])



# ==================================================
# Save tokens
# ==================================================

with open(TOKEN_OUTPUT,"w",encoding="utf8") as f:
    json.dump(tokens,f,indent=2,ensure_ascii=False)

print("Saved tokens:",TOKEN_OUTPUT)



# ==================================================
# MIDI
# ==================================================

midi=tokens_to_midi(tokens)
midi.save(MIDI_OUTPUT)
print("Saved MIDI:",MIDI_OUTPUT)