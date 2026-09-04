import os
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from build_jazz_dataset import JazzDataset
from Transformer import JazzTransformer
from torch.amp import autocast, GradScaler


# =========================
# Config
# =========================
BASE_DIR = "/content/drive/MyDrive/JazzGen_Data"
JSON_DIR = BASE_DIR + "/token"
VOCAB_FILE = BASE_DIR + "/vocabulary/vocabulary.json"
CHECKPOINT_DIR = "/content/drive/MyDrive/JazzGen_Data/check_point"
SEQ_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
os.makedirs(CHECKPOINT_DIR,exist_ok=True)

# Dataset
train_dataset = JazzDataset(json_dir=JSON_DIR,vocab_file=VOCAB_FILE,seq_length=SEQ_LENGTH,stride=256,split="train")
val_dataset = JazzDataset(json_dir=JSON_DIR,vocab_file=VOCAB_FILE,seq_length=SEQ_LENGTH,stride=256,split="val")
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,pin_memory=True,drop_last=True,num_workers=2)
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,pin_memory=True,num_workers=2)
print("Train samples:",len(train_dataset))
print("Validation samples:",len(val_dataset))

# Model
vocab_size = len(train_dataset.token_to_id)
model = JazzTransformer(vocab_size=vocab_size,max_seq_len=SEQ_LENGTH,d_model=512,n_heads=8,num_layers=8,dropout=0.1)
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


BEST_MODEL = (CHECKPOINT_DIR +"/best_model.pt")
start_epoch = 1
best_loss = float("inf")
if os.path.exists(BEST_MODEL):
    print("Loading checkpoint...")
    checkpoint = torch.load(BEST_MODEL,map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = (checkpoint["epoch"] + 1)
    best_loss = checkpoint["loss"]
    print("Resume epoch:",start_epoch)

def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        x = batch["input_ids"].to(DEVICE,non_blocking = True)
        y = batch["labels"].to(DEVICE, non_blocking = True)
        optimizer.zero_grad()
        with autocast("cuda"):
           logits = model(x)
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
        with autocast("cuda"):
           logits=model(x)
           loss=criterion(logits.reshape(-1,vocab_size),y.reshape(-1))
        total_loss+=loss.item()
    return total_loss/len(val_loader)

# =========================
# Main Training Loop
# =========================

def main():
    global best_loss
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
            torch.save({
            "epoch":epoch,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(), 
            "scheduler_state_dict":scheduler.state_dict(),
            "scaler_state_dict":scaler.state_dict(),
            "loss":val_loss},
            f"{CHECKPOINT_DIR}/best_model.pt")
            print("Saved best model")
    # 定期保存
        if epoch % 10 ==0:
            torch.save(model.state_dict(),f"{CHECKPOINT_DIR}/epoch_{epoch}.pt")
    print("Training Finished")

if __name__ == "__main__":
    main()