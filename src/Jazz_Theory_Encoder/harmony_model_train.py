import sys
import os
PROJECT_ROOT=os.path.dirname(os.path.dirname( os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from config import HARMONY_TOKEN_DIR,HARMONY_VOCAB_DIR,HARMONY_CHECKPOINT_DIR
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ireal_process.harmony_dataset import HarmonyDataset
from Jazz_Theory_Encoder.harmony_model import HarmonyModel

DEVICE=torch.device( "cuda" if torch.cuda.is_available()else "cpu")



class HarmonyTrainHead(nn.Module):

    def __init__( self,d_model,chord_vocab_size):
        super().__init__()
        self.fc=nn.Linear(d_model,chord_vocab_size)


    def forward(self,x):
        return self.fc(x)

# ==========================
# Dataset
# ==========================
dataset=HarmonyDataset(
        json_file=os.path.join(HARMONY_TOKEN_DIR,"ireal_harmony_form.json"),
        vocab_dir= HARMONY_VOCAB_DIR,
        seq_length=128)
loader=DataLoader(dataset,batch_size=16,shuffle=True)

# ==========================
# Encoder
# ==========================

encoder=HarmonyModel(
    chord_vocab_size=len(dataset.chord_vocab),
    duration_vocab_size=10,
    beat_vocab_size=20,
    section_vocab_size=20,
    time_vocab_size=10,
    d_model=512,
    n_heads=8,
    num_layers=6
)

encoder.to(DEVICE)

# ==========================
# Temporary Head
# ==========================
head=HarmonyTrainHead(d_model=512,chord_vocab_size=len(dataset.chord_vocab))
head.to(DEVICE)
# ==========================
# Optimizer
# ==========================
optimizer=torch.optim.AdamW(list(encoder.parameters())+list(head.parameters()),lr=1e-4,weight_decay=0.01)
criterion=nn.CrossEntropyLoss(ignore_index=-100)



# ==========================
# Train
# ==========================

EPOCHS=50
best_loss=999
for epoch in range(EPOCHS):
    encoder.train()
    head.train()
    total_loss=0
    for batch in loader:
        input_ids=batch["input_ids"].to(DEVICE)
        scale_vector=batch["scale_vector"].to(DEVICE)
        chord_tones=batch["chord_tones_vector"].to(DEVICE)
        guide=batch["guide_tones_vector"].to(DEVICE)
        tension=batch["tensions_vector"].to(DEVICE)
        available=batch["available_tensions_vector"].to(DEVICE)
        avoid=batch["avoid_vector"].to(DEVICE)
        target=batch["labels"].to(DEVICE)
        mask=batch[ "attention_mask"].to(DEVICE)
        # Encoder

        #debug
        # print("Max chord:", batch["input_ids"].max())
        # print("min chord:", batch["input_ids"].min())

        memory=encoder(
            input_ids,
            scale_vector,
            chord_tones,
            guide,
            tension,
            available,
            avoid,
            mask
        )
        # prediction
        logits=head(memory)

        #debug

        # print("logits:", logits.shape)
        # print("target:", target.shape)

        loss=criterion(logits.reshape(-1,len(dataset.chord_vocab)),target.reshape( -1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(),1.0)
        optimizer.step()
        total_loss+=loss.item()
    avg_loss=total_loss/len(loader)

    print(f"Epoch {epoch} Loss {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss=avg_loss
        torch.save(
            {
            "epoch":epoch,
            "loss":avg_loss,
            "model_state_dict":encoder.state_dict()
            },
            os.path.join(HARMONY_CHECKPOINT_DIR,"harmony_best.pt")
        )
