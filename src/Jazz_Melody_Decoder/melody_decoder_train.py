import os
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .build_jazz_dataset import JazzDataset
from .melody_decoder_Transformer import JazzTransformer
from src.Jazz_Theory_Encoder.harmony_model import HarmonyModel
from torch.amp import autocast, GradScaler
from src.config import MELODY_TOKEN_DIR,MELODY_VOCAB_FILE,MELODY_CHECKPOINT_DIR,HARMONY_CHECKPOINT_DIR


# =========================
# Config
# =========================
# BASE_DIR = "/content/drive/MyDrive/JazzGen_Data"
# JSON_DIR = BASE_DIR + "/token"
# VOCAB_FILE = BASE_DIR + "/vocabulary/vocabulary.json"
# CHECKPOINT_DIR = "/content/drive/MyDrive/JazzGen_Data/check_point"
SEQ_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 20
LR = 3e-4
# Early stopping
PATIENCE = 8
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
os.makedirs(MELODY_CHECKPOINT_DIR,exist_ok=True)

# Dataset
train_dataset = JazzDataset(solo_dir=MELODY_TOKEN_DIR,vocab_file=MELODY_VOCAB_FILE,seq_length=SEQ_LENGTH,stride=256,split="train")
val_dataset = JazzDataset(solo_dir=MELODY_TOKEN_DIR,vocab_file=MELODY_VOCAB_FILE,seq_length=SEQ_LENGTH,stride=256,split="val")
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,pin_memory=True,drop_last=True,num_workers=2)
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,pin_memory=True,num_workers=2)
print("Train samples:",len(train_dataset))
print("Validation samples:",len(val_dataset))

# Model
vocab_size = len(train_dataset.token_to_id)

harmony_encoder = HarmonyModel(
    chord_vocab_size=1064,
    duration_vocab_size=10,
    beat_vocab_size=20,
    section_vocab_size=20,
    time_vocab_size=10,
    d_model=512,
    n_heads=8,
    num_layers=6
)

checkpoint=torch.load(HARMONY_CHECKPOINT_DIR,map_location=DEVICE)
if "encoder" in checkpoint:
    state=checkpoint["encoder"]

else:
    state=checkpoint["model_state_dict"]

harmony_encoder.load_state_dict(state,strict=False)
harmony_encoder.to(DEVICE)
harmony_encoder.eval()
for p in harmony_encoder.parameters():
    p.requires_grad=False
print("Harmony Encoder loaded")



model = JazzTransformer(
    vocab_size=vocab_size,
    max_seq_len=SEQ_LENGTH,
    d_model=512,
    n_heads=8,
    num_layers=8,
    dropout=0.1)
print("Model vocab size:", vocab_size)
model.to(DEVICE)
# Loss
criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_id)
# Optimizer
optimizer = torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
# Train Function

scaler = GradScaler("cuda")


BEST_MODEL = (MELODY_CHECKPOINT_DIR +"/melody_best.pt")
start_epoch = 1
best_loss = float("inf")
best_epoch = 0
# if os.path.exists(MELODY_CHECKPOINT_DIR):
#     print("Loading checkpoint...")
#     checkpoint = torch.load(BEST_MODEL,map_location=DEVICE)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

#     if "scheduler_state_dict" in checkpoint:
#         scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

#     if "scaler_state_dict" in checkpoint:    
#        scaler.load_state_dict(checkpoint["scaler_state_dict"])

#     start_epoch = (checkpoint["epoch"] + 1)
#     best_loss = checkpoint["loss"]
#     best_epoch = checkpoint["epoch"]

#     print("Resume epoch:",start_epoch)

def train_one_epoch(epoch):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        x = batch["input_ids"].to(DEVICE,non_blocking = True)
        y = batch["labels"].to(DEVICE, non_blocking = True)
        harmony = {k:v.to(DEVICE,non_blocking = True) for k,v in batch["harmony"].items()}

        optimizer.zero_grad()

        with torch.no_grad():
            memory = harmony_encoder.encode(**harmony)

        with autocast("cuda"):
           logits = model(x,memory)
        # print( "input max:",x.max().item(),"input min:",x.min().item())
        # logits:
        # [batch, seq, vocab]
           loss = criterion(logits.reshape(-1, vocab_size),y.reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        if batch_idx % 50 == 0:
            gpu_memory = (torch.cuda.memory_allocated()/1024**3)
            print(f"Epoch {epoch} "
                  f"Step {batch_idx} "
                  f"Loss {loss.item():.4f}"
                  f"GPU {gpu_memory:.2f}GB")
            
    return total_loss / len(train_loader)

# Validation

@torch.no_grad()
def validate():
    model.eval()
    total_loss=0
    for batch in val_loader:
        x=batch["input_ids"].to(DEVICE,non_blocking = True)
        y=batch["labels"].to(DEVICE, non_blocking = True)
        harmony = {k:v.to(DEVICE,non_blocking = True) for k,v in batch["harmony"].items()}
        with autocast("cuda"):
           memory = harmony_encoder.encode(
               input_ids=harmony["input_ids"],
               scale_vector=harmony["scale_vector"],
               chord_tones_vector=harmony["chord_tones_vector"],guide_vector=harmony["guide_vector"],
               tension_vector=harmony["tension_vector"],available_tension_vector=harmony["available_tension_vector"],avoid_vector=harmony["avoid_vector"]
               )
           logits=model(x,memory)
           loss=criterion(logits.reshape(-1,vocab_size),y.reshape(-1))
        total_loss+=loss.item()
    return total_loss/len(val_loader)

# =========================
# Main Training Loop
# =========================

def main():
    global best_loss
    global best_epoch 

    patience_counter = 0
    for epoch in range(start_epoch,EPOCHS+1):
        train_loss=train_one_epoch(epoch)
        val_loss=validate()
        scheduler.step()
        print("="*50)
        print(f"""
          Epoch {epoch}
          Train Loss:{train_loss:.4f}
          Val Loss:{val_loss:.4f}
          LR:{optimizer.param_groups[0]['lr']}
""")
    # 保存最佳模型
        if val_loss < best_loss:
            best_loss=val_loss
            best_epoch = epoch
            patience_counter =0
            torch.save({
            "epoch":epoch,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(), 
            "scheduler_state_dict":scheduler.state_dict(),
            "scaler_state_dict":scaler.state_dict(),
            "loss":val_loss},
            BEST_MODEL)
            print("Saved best model")

        else:
            patience_counter += 1
            print(f"No improvement" f"{patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered")
                break
    # 定期保存
        if epoch % 10 ==0:
            torch.save(
                {
                    "epoch":epoch,
                    "model_state_dict":model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "loss":val_loss
                 },f"{MELODY_CHECKPOINT_DIR}/epoch_{epoch}.pt"
                )
        
        print("=" * 50)
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Val Loss: {best_loss:.4f}")


    print("Training Finished")

if __name__ == "__main__":
    main()