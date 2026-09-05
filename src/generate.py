import os
import json
import torch
import torch.nn.functional as F

from Transformer import JazzTransformer


# =========================
# Config
# =========================

BASE_DIR = "/content/drive/MyDrive/JazzGen_Data"
VOCAB_FILE = (BASE_DIR + "/vocabulary/vocabulary.json")
CHECKPOINT = (BASE_DIR +"/check_point/best_model.pt")
OUTPUT_DIR = (BASE_DIR +"/generated")
os.makedirs(OUTPUT_DIR,exist_ok=True)
SEQ_LENGTH = 512
MAX_GENERATE_LENGTH = 2048
TEMPERATURE = 0.8
TOP_K = 50

REPETITION_PENALTY = 1.1
DEVICE = torch.device("cuda"if torch.cuda.is_available() else "cpu")


# =========================
# Load Vocabulary
# =========================

with open(VOCAB_FILE,"r") as f:
    vocab = json.load(f)

token_to_id = vocab["token_to_id"]
id_to_token = {int(k):v for k,v in vocab["id_to_token"].items()}

BOS_ID = token_to_id["<BOS>"]
EOS_ID = token_to_id["<EOS>"]
PAD_ID = token_to_id["<PAD>"]
UNK_ID = token_to_id["<UNK>"]

VOCAB_SIZE = len(token_to_id)



# =========================
# Load Model
# =========================

model = JazzTransformer(
    vocab_size=VOCAB_SIZE,
    max_seq_len=SEQ_LENGTH,
    d_model=512,
    n_heads=8,
    num_layers=8,
    dropout=0.1
)

checkpoint = torch.load(CHECKPOINT,map_location=DEVICE,weights_only=False)

model.load_state_dict(checkpoint["model_state_dict"])

model.to(DEVICE)

model.eval()

print("=" * 50)

print("Loaded checkpoint epoch:",checkpoint["epoch"])

print("Vocabulary:",VOCAB_SIZE)

print("=" * 50)



# =========================
# Sampling Function
# =========================

def sample_next_token(logits,history_tokens):

    logits = logits / TEMPERATURE

    logits[PAD_ID] = -float("inf")
    logits[UNK_ID] = -float("inf")

    recent_tokens = set(history_tokens[-50:])


    for token in recent_tokens:
        logits[token] /= REPETITION_PENALTY

    # top-k filtering
    values, indices = torch.topk(logits,TOP_K)

    probs = F.softmax(values,dim=-1)
    next_index = torch.multinomial(probs,num_samples=1)
    next_token = indices[next_index]
    return next_token.item()



# =========================
# Generate
# =========================


@torch.no_grad()
def generate():

    tokens = [BOS_ID]
    for step in range(MAX_GENERATE_LENGTH):
        input_ids = torch.tensor(tokens,dtype=torch.long).unsqueeze(0)
        input_ids = input_ids.to(DEVICE)
        logits = model(input_ids)
        next_logits = logits[0,-1]
        next_token = sample_next_token(next_logits,tokens)
        tokens.append(next_token)
        if next_token == EOS_ID:
            print("EOS generated at step:", step)
    return tokens



# =========================
# Decode Token
# =========================


def decode_tokens(tokens):
    result=[]
    for t in tokens:
        result.append(id_to_token[t])
    return result



# =========================
# Main
# =========================

if __name__ == "__main__":

    generated_tokens = generate()
    print( "Generated length:",len(generated_tokens))
    decoded = decode_tokens(generated_tokens)

    output_file = (OUTPUT_DIR +"/generated_tokens.json")
    with open(output_file,"w") as f:
        json.dump(decoded,f,indent=2)
    print("Saved:",output_file)
    print("\nFirst tokens:")
    print(decoded[:100])