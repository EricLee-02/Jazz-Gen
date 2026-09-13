import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .jazz_generation_dataset import JazzGenerationDataset
from .Jazz_Theory_Encoder.harmony_model import HarmonyModel
from .Jazz_Melody_Decoder.melody_decoder_Transformer import JazzTransformer
from .melody_generate_model import JazzGenerationModel
from .config import MELODY_TOKEN_DIR,MELODY_VOCAB_DIR,HARMONY_CHECKPOINT_DIR,MELODY_VOCAB_FILE



# ===============================
# Config
# ===============================

BASE_DIR="/Volumes/My Passport/Jazz Gen"
TOKEN_DIR = (BASE_DIR+ "/data/processed/jazz_json_v2_token/token")
VOCAB_FILE = (BASE_DIR+"/data/processed/jazz_json_v2_token/vocabulary/vocabulary.json")
HARMONY_CHECKPOINT = (BASE_DIR+ "/check_point/best_harmony.pt")
SAVE_DIR = (BASE_DIR+"/check_point/generation")
os.makedirs(SAVE_DIR,exist_ok=True)
DEVICE=torch.device("cuda"if torch.cuda.is_available()else "cpu")
BATCH_SIZE=8
EPOCHS=50



# ===============================
# Dataset
# ===============================


dataset=JazzGenerationDataset(solo_dir=MELODY_TOKEN_DIR,vocab_file=MELODY_VOCAB_FILE,seq_length=512)
loader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)



print("Dataset:",len(dataset))

# ===============================
# Harmony Encoder
# ===============================


harmony_model=HarmonyModel(
    chord_vocab_size=1064,
    duration_vocab_size=10,
    beat_vocab_size=20,
    section_vocab_size=20,
    time_vocab_size=10,
    d_model=512,
    n_heads=8,
    num_layers=6
)


checkpoint=torch.load(HARMONY_CHECKPOINT_DIR, map_location=DEVICE)
if "encoder" in checkpoint:
    state = checkpoint["encoder"]
elif "model_state_dict" in checkpoint:
    state = checkpoint["model_state_dict"]
else:
    raise Exception("Wrong checkpoint")


harmony_model.load_state_dict(state,strict=False)
print("Harmony loaded")


for param in harmony_model.parameters():
    param.requires_grad = False
harmony_model.eval()

with open(MELODY_VOCAB_FILE,"r") as f :
    vocab = json.load(f)

vocab_size = len(vocab["token_to_id"])
# ===============================
# Melody Decoder
# ===============================
melody_decoder=JazzTransformer(
    vocab_size=vocab_size,
    max_seq_len=512,
    d_model=512,
    n_heads=8,
    num_layers=8,
    dropout=0.1
)



# ===============================
# Full Model
# ===============================

model=JazzGenerationModel( harmony_model,melody_decoder)

model.to(DEVICE)



# ===============================
# Optimizer
# Differential LR
# ===============================

optimizer=torch.optim.AdamW(model.melody_decoder.parameters(),lr=1e-4,weight_decay=0.01)

# ===============================
# Scheduler
# ===============================
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
# ===============================
# Loss
# ===============================
criterion=nn.CrossEntropyLoss(ignore_index=-100)
# ===============================
# AMP
# ===============================
scaler=torch.cuda.amp.GradScaler()

# ===============================
# Train
# ===============================

def train_epoch(epoch):
    model.train()
    total_loss=0
    for step,batch in enumerate(loader):
        harmony=batch["harmony"]
        for k in harmony:
            harmony[k]=(harmony[k].to(DEVICE))
        melody_input=batch["melody_input"].to(DEVICE)
        melody_target=batch["melody_target"].to(DEVICE)
        optimizer.zero_grad()
        with torch.autocast(device_type=DEVICE.type,dtype=torch.float16):
            logits=model(harmony, melody_input)
            loss=criterion(logits.reshape(-1,logits.size(-1)),melody_target.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(optimizer )
        scaler.update()
        total_loss+=loss.item()
        if step%20==0:
            print(f"Epoch {epoch}Step {step}/{len(loader)}Loss {loss.item():.4f}")
    scheduler.step()
    return total_loss/len(loader)

# ===============================
# Main
# ===============================

best_loss=float("inf")

for epoch in range(EPOCHS):
    loss=train_epoch(epoch)
    print("Epoch:",epoch, "Average Loss:", loss)
    torch.save({
            "epoch":epoch,
            "loss":loss,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict()
    },
           os.path.join(SAVE_DIR,"generation_last.pt")
    )
    if loss < best_loss:
        best_loss=loss
        torch.save(
            {
                "epoch":epoch,
                "loss":loss,
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict()
            },
            os.path.join(SAVE_DIR,"generation_best.pt")
        )
        print("Saved best model")
