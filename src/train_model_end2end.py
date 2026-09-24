# import os
# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader

# from .jazz_generation_dataset import JazzGenerationDataset
# from .Jazz_Theory_Encoder.harmony_model import HarmonyModel
# from .Jazz_Melody_Decoder.melody_decoder_Transformer import JazzTransformer
# from .config import (
#     MELODY_TOKEN_DIR,
#     MELODY_CHECKPOINT_FILE,
#     HARMONY_CHECKPOINT_FILE,
#     MELODY_VOCAB_FILE,
#     GENERATION_CHECKPOINT_DIR,
#     MELODY_MAX_SEQ_LEN,
#     MELODY_D_MODEL,
#     MELODY_HEADS,
#     MELODY_LAYERS,
#     DROPOUT,
#     CHORD_VOCAB_SIZE,
#     DURATION_VOCAB_SIZE,
#     BEAT_VOCAB_SIZE,
#     SECTION_VOCAB_SIZE,
#     TIME_VOCAB_SIZE,
#     D_MODEL,
#     NUM_HEADS,
#     NUM_LAYERS,
#     EPOCHS,
#     TRAIN_RATIO,
#     GENERATE_BATCH_SIZE,
#     ACCUM_STEPS,
#     PATIENCE,
#     WEIGHT_DECAY,
#     MIN_DELTA,
#     SEED,

#     HARMONY_WEIGHT,
#     MELODY_WEIGHT
# )

# os.makedirs(GENERATION_CHECKPOINT_DIR, exist_ok=True)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# HARMONY_LR = 1e-5
# MELODY_LR = 3e-5

# print("Device:", DEVICE)

# # ==================================================
# # Dataset
# # ==================================================

# train_dataset = JazzGenerationDataset(
#     solo_dir=MELODY_TOKEN_DIR,
#     vocab_file=MELODY_VOCAB_FILE,
#     seq_length=MELODY_MAX_SEQ_LEN,
#     max_harmony_len=128,
#     stride=256,
#     split="train",
#     train_ratio=TRAIN_RATIO,
#     seed=SEED,
# )

# val_dataset = JazzGenerationDataset(
#     solo_dir=MELODY_TOKEN_DIR,
#     vocab_file=MELODY_VOCAB_FILE,
#     seq_length=MELODY_MAX_SEQ_LEN,
#     max_harmony_len=128,
#     stride=256,
#     split="val",
#     train_ratio=TRAIN_RATIO,
#     seed=SEED,
# )

# print("Train samples:", len(train_dataset))
# print("Validation samples:", len(val_dataset))

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=GENERATE_BATCH_SIZE,
#     shuffle=True,
#     num_workers=0,
#     pin_memory=(DEVICE.type == "cuda"),
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=GENERATE_BATCH_SIZE,
#     shuffle=False,
#     num_workers=0,
#     pin_memory=(DEVICE.type == "cuda"),
# )

# # ==================================================
# # Vocabulary
# # ==================================================

# with open(MELODY_VOCAB_FILE, "r") as f:
#     vocab = json.load(f)

# vocab_size = len(vocab["token_to_id"])
# print("Melody vocab size:", vocab_size)

# # ==================================================
# # Harmony Encoder
# # ==================================================

# harmony_model = HarmonyModel(
#     chord_vocab_size=CHORD_VOCAB_SIZE,
#     duration_vocab_size=DURATION_VOCAB_SIZE,
#     beat_vocab_size=BEAT_VOCAB_SIZE,
#     section_vocab_size=SECTION_VOCAB_SIZE,
#     time_vocab_size=TIME_VOCAB_SIZE,
#     d_model=D_MODEL,
#     num_heads=NUM_HEADS,
#     num_layers=NUM_LAYERS,
# )

# harmony_checkpoint = torch.load(
#     HARMONY_CHECKPOINT_FILE,
#     map_location="cpu",
# )

# if "encoder" in harmony_checkpoint:
#     harmony_state = harmony_checkpoint["encoder"]
# elif "model_state_dict" in harmony_checkpoint:
#     harmony_state = harmony_checkpoint["model_state_dict"]
# else:
#     harmony_state = harmony_checkpoint

# harmony_result = harmony_model.load_state_dict(
#     harmony_state,
#     strict=False,
# )

# print("Harmony Encoder loaded")
# print("Harmony missing keys:", harmony_result.missing_keys)
# print("Harmony unexpected keys:", harmony_result.unexpected_keys)

# # Unfreeze encoder for end-to-end joint fine-tuning
# for param in harmony_model.parameters():
#     param.requires_grad = True

# harmony_model.to(DEVICE)

# # ==================================================
# # Melody Decoder
# # ==================================================

# melody_decoder = JazzTransformer(
#     vocab_size=vocab_size,
#     max_seq_len=MELODY_MAX_SEQ_LEN,
#     d_model=MELODY_D_MODEL,
#     num_heads=MELODY_HEADS,
#     num_layers=MELODY_LAYERS,
#     dropout=DROPOUT,
# )

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
#     strict=False,
# )

# print("Melody Decoder loaded")
# print("Melody missing keys:", melody_result.missing_keys)
# print("Melody unexpected keys:", melody_result.unexpected_keys)

# if isinstance(melody_checkpoint, dict):
#     if "epoch" in melody_checkpoint:
#         print("Previous Melody epoch:", melody_checkpoint["epoch"])
#     if "loss" in melody_checkpoint:
#         print("Previous Melody loss:", melody_checkpoint["loss"])

# melody_decoder.to(DEVICE)

# # ==================================================
# # Optimizer: Harmony Encoder + Melody Decoder
# # ==================================================

# optimizer = torch.optim.AdamW(
#     [
#         {
#             "params": harmony_model.parameters(),
#             "lr": HARMONY_LR,
#         },
#         {
#             "params": melody_decoder.parameters(),
#             "lr": MELODY_LR,
#         },
#     ],
#     weight_decay=WEIGHT_DECAY,
# )

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=EPOCHS,)

# melody_criterion = nn.CrossEntropyLoss(ignore_index=-100)
# harmony_criterion = nn.CrossEntropyLoss(ignore_index=-100)
# scaler = torch.amp.GradScaler( "cuda", enabled=(DEVICE.type == "cuda"),)

# trainable_params = ( list(harmony_model.parameters()) + list(melody_decoder.parameters()))

# # ==================================================
# # Train
# # ==================================================

# def train_epoch(epoch):
#     harmony_model.train()
#     melody_decoder.train()
#     total_loss = 0.0
#     optimizer.zero_grad(set_to_none=True)
#     for step, batch in enumerate(train_loader):
#         harmony = { k: v.to(DEVICE, non_blocking=True)  for k, v in batch["harmony"].items() }
#         melody_input = batch["melody_input"].to( DEVICE, non_blocking=True, )
#         melody_target = batch["melody_target"].to( DEVICE,non_blocking=True,)

#         with torch.autocast(device_type=DEVICE.type,dtype=torch.float16,enabled=(DEVICE.type == "cuda"), ):
#             # No torch.no_grad(): gradients must flow
#             # through Decoder and Harmony Encoder.
#             memory,harmony_logits = harmony_model.encode(**harmony)
#             melody_logits = melody_decoder( melody_input,memory)
#             melody_loss = melody_criterion(melody_logits.reshape(-1,melody_logits.size(-1)), melody_target.reshape(-1))
#             harmony_target = harmony["input_ids"]
#             harmony_target = harmony_target.masked_fill(harmony["attention_mask"] == 0, -100)
#             harmony_loss = harmony_criterion(harmony_logits.reshape(-1,harmony_logits.size(-1)),harmony_target.reshape(-1))
#             loss = MELODY_WEIGHT * melody_loss +HARMONY_WEIGHT * harmony_loss
#             scaled_loss = loss / ACCUM_STEPS
#         scaler.scale(scaled_loss).backward()
#         should_step = ((step + 1) % ACCUM_STEPS == 0 or step + 1 == len(train_loader))

#         if should_step:
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_( trainable_params,max_norm=1.0,)
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad(set_to_none=True)
#         total_loss += loss.item()

#         if step % 20 == 0:
#             gpu_mem = (torch.cuda.memory_allocated() / 1024 ** 3 if DEVICE.type == "cuda" else 0.0)
#             print(
#                 f"Epoch {epoch + 1} "
#                 f"Step {step}/{len(train_loader)} "
#                 f"Loss {loss.item():.4f} "
#                 f"H-LR {optimizer.param_groups[0]['lr']:.2e} "
#                 f"M-LR {optimizer.param_groups[1]['lr']:.2e} "
#                 f"GPU {gpu_mem:.2f}GB"
#             )
#     return total_loss / len(train_loader)

# # ==================================================
# # Validation
# # ==================================================

# @torch.no_grad()
# def validate_epoch():
#     harmony_model.eval()
#     melody_decoder.eval()
#     total_loss = 0.0
#     for batch in val_loader:
#         harmony = {k: v.to(DEVICE, non_blocking=True) for k, v in batch["harmony"].items() }
#         melody_input = batch["melody_input"].to(DEVICE,non_blocking=True, )
#         melody_target = batch["melody_target"].to( DEVICE, non_blocking=True,)

#         with torch.autocast( device_type=DEVICE.type,dtype=torch.float16, enabled=(DEVICE.type == "cuda"),):
#             memory,harmony_logits = harmony_model.encode(**harmony)
#             melody_logits = melody_decoder( melody_input, memory,)
#             melody_loss = melody_criterion(melody_logits.reshape(-1,melody_logits.size(-1)), melody_target.reshape(-1))
#             harmony_target = harmony["input_ids"]
#             harmony_target = harmony_target.masked_fill(harmony["attention_mask"] == 0, -100)
#             harmony_loss = harmony_criterion(harmony_logits.reshape(-1,harmony_logits.size(-1)),harmony_target.reshape(-1))
#             loss = MELODY_WEIGHT * melody_loss +HARMONY_WEIGHT * harmony_loss
#         total_loss += loss.item()
#     return total_loss / len(val_loader)

# # ==================================================
# # Checkpoint
# # ==================================================

# def build_checkpoint(epoch, train_loss, val_loss):
#     return {
#         "epoch": epoch,
#         "train_loss": train_loss,
#         "val_loss": val_loss,
#         "harmony_state_dict": harmony_model.state_dict(),
#         "melody_state_dict": melody_decoder.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "scheduler_state_dict": scheduler.state_dict(),
#         "scaler_state_dict": scaler.state_dict(),
#         "vocab_size": vocab_size,
#         "token_to_id": vocab["token_to_id"],
#         "harmony_lr": optimizer.param_groups[0]["lr"],
#         "melody_lr": optimizer.param_groups[1]["lr"],
#     }

# # ==================================================
# # Main
# # ==================================================

# def main():
#     best_val_loss = float("inf")
#     patience_counter = 0
#     for epoch in range(EPOCHS):
#         print(f"\n========== Epoch {epoch + 1}/{EPOCHS} ==========" )
#         train_loss = train_epoch(epoch)
#         val_loss = validate_epoch()
#         # Only once per epoch
#         scheduler.step()
#         print(f"\nEpoch {epoch + 1} finished")
#         print(f"Train Loss: {train_loss:.4f}")
#         print(f"Val Loss  : {val_loss:.4f}")
#         print( f"Harmony LR: " f"{optimizer.param_groups[0]['lr']:.8f}")
#         print( f"Melody LR : "f"{optimizer.param_groups[1]['lr']:.8f}")
#         checkpoint = build_checkpoint(epoch, train_loss, val_loss, )
#         torch.save(checkpoint, os.path.join(  GENERATION_CHECKPOINT_DIR, "generation_last.pt",),)
#         if val_loss < best_val_loss - MIN_DELTA:
#             best_val_loss = val_loss
#             patience_counter = 0
#             torch.save(  checkpoint, os.path.join(  GENERATION_CHECKPOINT_DIR,  "generation_best.pt", ), )
#             print( f"Saved best model " f"(Val Loss {val_loss:.4f})")
#         else:
#             patience_counter += 1
#             print( f"No improvement: " f"{patience_counter}/{PATIENCE}")
#             if patience_counter >= PATIENCE:
#                 print("\nEarly stopping triggered.")
#                 print(  f"Best Val Loss: " f"{best_val_loss:.4f}")
#                 break

#     print("\nTraining finished.")
# if __name__ == "__main__":
#     main()



import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .jazz_generation_dataset import JazzGenerationDataset
from .Jazz_Theory_Encoder.harmony_model import HarmonyModel
from .Jazz_Melody_Decoder.melody_decoder_Transformer import JazzTransformer
from .config import (
    MELODY_TOKEN_DIR,
    MELODY_CHECKPOINT_FILE,
    HARMONY_CHECKPOINT_FILE,
    MELODY_VOCAB_FILE,
    GENERATION_CHECKPOINT_DIR,

    MELODY_MAX_SEQ_LEN,
    MELODY_D_MODEL,
    MELODY_HEADS,
    MELODY_LAYERS,
    DROPOUT,

    CHORD_VOCAB_SIZE,
    DURATION_VOCAB_SIZE,
    BEAT_VOCAB_SIZE,
    SECTION_VOCAB_SIZE,
    TIME_VOCAB_SIZE,

    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,

    EPOCHS,
    TRAIN_RATIO,
    GENERATE_BATCH_SIZE,
    ACCUM_STEPS,
    PATIENCE,
    WEIGHT_DECAY,
    MIN_DELTA,
    SEED,
)


# ==================================================
# Basic config
# ==================================================

os.makedirs(GENERATION_CHECKPOINT_DIR,exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Joint fine-tuning:
# Harmony Encoder uses a smaller LR
# to avoid destroying the pretrained representation.
HARMONY_LR = 5e-6
MELODY_LR = 3e-5
print("Device:", DEVICE)


# ==================================================
# Dataset
# ==================================================
train_dataset = JazzGenerationDataset(
    solo_dir=MELODY_TOKEN_DIR,
    vocab_file=MELODY_VOCAB_FILE,
    seq_length=MELODY_MAX_SEQ_LEN,
    max_harmony_len=512,
    stride=256,
    split="train",
    train_ratio=TRAIN_RATIO,
    seed=SEED,
)
val_dataset = JazzGenerationDataset(
    solo_dir=MELODY_TOKEN_DIR,
    vocab_file=MELODY_VOCAB_FILE,
    seq_length=MELODY_MAX_SEQ_LEN,
    max_harmony_len=512,
    stride=256,
    split="val",
    train_ratio=TRAIN_RATIO,
    seed=SEED,
)

print( "Train samples:", len(train_dataset))
print( "Validation samples:", len(val_dataset))


train_loader = DataLoader(
    train_dataset,
    batch_size=GENERATE_BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(
        DEVICE.type == "cuda"
    ),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=GENERATE_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(
        DEVICE.type == "cuda"
    ),
)


# ==================================================
# Vocabulary
# ==================================================

with open( MELODY_VOCAB_FILE, "r", encoding="utf8") as f:
    vocab = json.load(f)

token_to_id = vocab["token_to_id"]

id_to_token = {int(k): v for k, v in vocab["id_to_token"].items()}
vocab_size = len(token_to_id)
print( "Melody vocab size:",vocab_size)


# ==================================================
# Harmony Encoder
# ==================================================

harmony_model = HarmonyModel(
    chord_vocab_size=CHORD_VOCAB_SIZE,
    duration_vocab_size=DURATION_VOCAB_SIZE,
    beat_vocab_size=BEAT_VOCAB_SIZE,
    section_vocab_size=SECTION_VOCAB_SIZE,
    time_vocab_size=TIME_VOCAB_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
)

harmony_checkpoint = torch.load( HARMONY_CHECKPOINT_FILE, map_location="cpu",)
if "encoder" in harmony_checkpoint:
    harmony_state = (harmony_checkpoint[ "encoder"])

elif "model_state_dict" in harmony_checkpoint:
    harmony_state = (harmony_checkpoint["model_state_dict"])

else:
    harmony_state = ( harmony_checkpoint)

harmony_result = (harmony_model.load_state_dict(harmony_state,strict=False,))


print("Harmony Encoder loaded")
print( "Harmony missing keys:",harmony_result.missing_keys)
print("Harmony unexpected keys:",harmony_result.unexpected_keys)
# Do not silently train a partially-loaded encoder.
if (
    harmony_result.missing_keys
    or
    harmony_result.unexpected_keys
):

    raise RuntimeError(
        "Harmony checkpoint does not "
        "exactly match HarmonyModel. "
        "Please fix missing/unexpected "
        "keys before joint fine-tuning."
    )


# ==================================================
# Unfreeze Harmony Encoder
# ==================================================

for param in (
    harmony_model.parameters()
):

    param.requires_grad = True


harmony_model.to(
    DEVICE
)


# ==================================================
# Melody Decoder
# ==================================================

melody_decoder = JazzTransformer(
    vocab_size=vocab_size,
    max_seq_len=MELODY_MAX_SEQ_LEN,
    d_model=MELODY_D_MODEL,
    num_heads=MELODY_HEADS,
    num_layers=MELODY_LAYERS,
    dropout=DROPOUT,
)


melody_checkpoint = torch.load(
    MELODY_CHECKPOINT_FILE,
    map_location="cpu",
)


if "model_state_dict" in melody_checkpoint:

    melody_state = (
        melody_checkpoint[
            "model_state_dict"
        ]
    )

elif "decoder" in melody_checkpoint:

    melody_state = (
        melody_checkpoint[
            "decoder"
        ]
    )

else:

    melody_state = (
        melody_checkpoint
    )


melody_result = (
    melody_decoder.load_state_dict(
        melody_state,
        strict=False,
    )
)


print(
    "Melody Decoder loaded"
)

print(
    "Melody missing keys:",
    melody_result.missing_keys
)

print(
    "Melody unexpected keys:",
    melody_result.unexpected_keys
)


if (
    melody_result.missing_keys
    or
    melody_result.unexpected_keys
):

    raise RuntimeError(
        "Melody checkpoint does not "
        "exactly match current decoder "
        "or vocabulary."
    )


if isinstance(
    melody_checkpoint,
    dict
):

    if "epoch" in melody_checkpoint:

        print(
            "Previous Melody epoch:",
            melody_checkpoint[
                "epoch"
            ]
        )

    if "loss" in melody_checkpoint:

        print(
            "Previous Melody loss:",
            melody_checkpoint[
                "loss"
            ]
        )


melody_decoder.to(
    DEVICE
)


# ==================================================
# Trainable params
# ==================================================

encoder_params = [
    p
    for p in
    harmony_model.parameters()
    if p.requires_grad
]

decoder_params = [
    p
    for p in
    melody_decoder.parameters()
    if p.requires_grad
]

trainable_params = (
    encoder_params
    +
    decoder_params
)


print(
    "Trainable Harmony params:",
    sum(
        p.numel()
        for p in
        encoder_params
    )
)

print(
    "Trainable Melody params:",
    sum(
        p.numel()
        for p in
        decoder_params
    )
)

print(
    "Total trainable params:",
    sum(
        p.numel()
        for p in
        trainable_params
    )
)


# ==================================================
# Optimizer
# ==================================================

optimizer = torch.optim.AdamW(
    [
        {
            "params":
                encoder_params,

            "lr":
                HARMONY_LR,
        },
        {
            "params":
                decoder_params,

            "lr":
                MELODY_LR,
        },
    ],

    weight_decay=WEIGHT_DECAY,
)


scheduler = (
    torch.optim.lr_scheduler
    .CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )
)


criterion = (
    nn.CrossEntropyLoss(
        ignore_index=-100
    )
)


scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(
        DEVICE.type == "cuda"
    ),
)


# ==================================================
# Preflight
# ==================================================

def preflight_check():

    print(
        "\n========== "
        "PREFLIGHT CHECK "
        "=========="
    )

    batch = next(
        iter(train_loader)
    )


    # ==================================
    # Count melody notes
    # ==================================

    pitch_ids = torch.tensor(
        [
            idx

            for token, idx
            in token_to_id.items()

            if token.startswith(
                "PITCH_"
            )
        ],

        dtype=torch.long,
    )


    melody_cpu = (
        batch[
            "melody_input"
        ]
    )


    pitch_mask = (
        melody_cpu.unsqueeze(-1)
        ==
        pitch_ids.view(
            1,
            1,
            -1
        )
    ).any(
        dim=-1
    )


    melody_note_counts = (
        pitch_mask.sum(
            dim=1
        )
    )


    harmony_note_counts = (
        batch[
            "harmony"
        ][
            "attention_mask"
        ]
        .sum(dim=1)
        .cpu()
    )


    print(
        "Melody note counts :",
        melody_note_counts.tolist()
    )

    print(
        "Harmony event counts:",
        harmony_note_counts.tolist()
    )


    if not torch.equal(
        melody_note_counts.cpu(),
        harmony_note_counts,
    ):

        raise RuntimeError(
            "Preflight failed: "
            "melody note count != "
            "harmony event count."
        )


    # ==================================
    # Check grammar
    # ==================================

    first_ids = (
        melody_cpu[0]
        .tolist()
    )

    preview_tokens = []


    for idx in first_ids:

        if (
            idx
            ==
            token_to_id[
                "<PAD>"
            ]
        ):

            break

        preview_tokens.append(
            id_to_token.get(
                idx,
                "<UNK_ID>"
            )
        )

        if (
            len(
                preview_tokens
            )
            >= 30
        ):

            break


    print(
        "Token preview:"
    )

    print(
        preview_tokens
    )


    expected_prefixes = [
        "SECTION_",
        "BAR_",
        "PERIOD_",
        "BEAT_",
        "DIVISION_",
        "TATUM_",
        "PITCH_",
        "DURATION_",
        "VELOCITY_",
    ]


    if (
        len(
            preview_tokens
        )
        >= 12
    ):

        first_note = (
            preview_tokens[
                3:12
            ]
        )


        if not all(
            token.startswith(
                prefix
            )

            for token, prefix
            in zip(
                first_note,
                expected_prefixes
            )
        ):

            raise RuntimeError(
                "Preflight failed: "
                "wrong note grammar.\n"
                f"Observed: {first_note}"
            )


    # ==================================
    # One forward pass
    # ==================================

    harmony = {
        k:
            v.to(
                DEVICE,
                non_blocking=True
            )

        for k, v
        in batch[
            "harmony"
        ].items()
    }


    melody_input = (
        batch[
            "melody_input"
        ]
        .to(
            DEVICE,
            non_blocking=True
        )
    )


    melody_target = (
        batch[
            "melody_target"
        ]
        .to(
            DEVICE,
            non_blocking=True
        )
    )


    harmony_model.eval()
    melody_decoder.eval()


    with torch.no_grad():

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=(
                DEVICE.type
                ==
                "cuda"
            ),
        ):

            memory = (
                harmony_model.encode(
                    **harmony
                )
            )


            if not torch.is_tensor(
                memory
            ):

                raise RuntimeError(
                    "HarmonyModel.encode() "
                    "did not return a Tensor. "
                    f"Got: {type(memory)}"
                )


            melody_logits = (
                melody_decoder(
                    melody_input,
                    memory,
                )
            )


            loss = (
                criterion(
                    melody_logits.reshape(
                        -1,
                        melody_logits.size(
                            -1
                        )
                    ),

                    melody_target.reshape(
                        -1
                    ),
                )
            )


    print(
        "Memory shape:",
        tuple(
            memory.shape
        )
    )

    print(
        "Melody logits shape:",
        tuple(
            melody_logits.shape
        )
    )

    print(
        "Initial preflight CE loss:",
        float(
            loss.item()
        )
    )


    harmony_model.train()
    melody_decoder.train()


    print(
        "PREFLIGHT PASSED"
    )

    print(
        "=====================================\n"
    )


# ==================================================
# Train
# ==================================================

def train_epoch(
    epoch
):

    harmony_model.train()
    melody_decoder.train()

    total_loss = 0.0

    optimizer.zero_grad(
        set_to_none=True
    )


    for step, batch in enumerate(
        train_loader
    ):

        harmony = {
            k:
                v.to(
                    DEVICE,
                    non_blocking=True
                )

            for k, v
            in batch[
                "harmony"
            ].items()
        }


        melody_input = (
            batch[
                "melody_input"
            ]
            .to(
                DEVICE,
                non_blocking=True
            )
        )


        melody_target = (
            batch[
                "melody_target"
            ]
            .to(
                DEVICE,
                non_blocking=True
            )
        )


        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=(
                DEVICE.type
                ==
                "cuda"
            ),
        ):

            # Melody loss backpropagates
            # through cross-attention
            # into Harmony Encoder.
            memory = (
                harmony_model.encode(
                    **harmony
                )
            )


            melody_logits = (
                melody_decoder(
                    melody_input,
                    memory,
                )
            )


            loss = (
                criterion(
                    melody_logits.reshape(
                        -1,
                        melody_logits.size(
                            -1
                        )
                    ),

                    melody_target.reshape(
                        -1
                    ),
                )
            )


            scaled_loss = (
                loss
                /
                ACCUM_STEPS
            )


        scaler.scale(
            scaled_loss
        ).backward()


        should_step = (
            (
                step + 1
            )
            %
            ACCUM_STEPS
            ==
            0

            or

            (
                step + 1
                ==
                len(
                    train_loader
                )
            )
        )


        if should_step:

            scaler.unscale_(
                optimizer
            )


            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=1.0,
            )


            scaler.step(
                optimizer
            )

            scaler.update()


            optimizer.zero_grad(
                set_to_none=True
            )


        total_loss += (
            loss.item()
        )


        if (
            step % 20
            ==
            0
        ):

            gpu_mem = (
                torch.cuda
                .memory_allocated()
                /
                1024 ** 3

                if
                DEVICE.type
                ==
                "cuda"

                else
                0.0
            )


            print(
                f"Epoch {epoch + 1} "
                f"Step {step}/{len(train_loader)} "
                f"MelodyLoss {loss.item():.4f} "
                f"H-LR {optimizer.param_groups[0]['lr']:.2e} "
                f"M-LR {optimizer.param_groups[1]['lr']:.2e} "
                f"GPU {gpu_mem:.2f}GB"
            )


    return (
        total_loss
        /
        len(
            train_loader
        )
    )


# ==================================================
# Validation
# ==================================================

@torch.no_grad()
def validate_epoch():

    harmony_model.eval()
    melody_decoder.eval()

    total_loss = 0.0


    for batch in val_loader:

        harmony = {
            k:
                v.to(
                    DEVICE,
                    non_blocking=True
                )

            for k, v
            in batch[
                "harmony"
            ].items()
        }


        melody_input = (
            batch[
                "melody_input"
            ]
            .to(
                DEVICE,
                non_blocking=True
            )
        )


        melody_target = (
            batch[
                "melody_target"
            ]
            .to(
                DEVICE,
                non_blocking=True
            )
        )


        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=(
                DEVICE.type
                ==
                "cuda"
            ),
        ):

            memory = (
                harmony_model.encode(
                    **harmony
                )
            )


            melody_logits = (
                melody_decoder(
                    melody_input,
                    memory,
                )
            )


            loss = (
                criterion(
                    melody_logits.reshape(
                        -1,
                        melody_logits.size(
                            -1
                        )
                    ),

                    melody_target.reshape(
                        -1
                    ),
                )
            )


        total_loss += (
            loss.item()
        )


    return (
        total_loss
        /
        len(
            val_loader
        )
    )


# ==================================================
# Combined checkpoint
# ==================================================

def combined_model_state_dict():

    state = {}


    for key, value in (
        harmony_model
        .state_dict()
        .items()
    ):

        state[
            f"harmony_encoder.{key}"
        ] = value


    for key, value in (
        melody_decoder
        .state_dict()
        .items()
    ):

        state[
            f"melody_decoder.{key}"
        ] = value


    return state


def build_checkpoint(
    epoch,
    train_loss,
    val_loss
):

    return {

        "epoch":
            epoch,

        "train_loss":
            train_loss,

        "val_loss":
            val_loss,


        # Individual models
        "harmony_state_dict":
            harmony_model.state_dict(),

        "melody_state_dict":
            melody_decoder.state_dict(),


        # Combined model
        "model_state_dict":
            combined_model_state_dict(),


        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),


        "vocab_size":
            vocab_size,

        "token_to_id":
            token_to_id,


        "harmony_lr":
            optimizer.param_groups[
                0
            ][
                "lr"
            ],

        "melody_lr":
            optimizer.param_groups[
                1
            ][
                "lr"
            ],
    }


# ==================================================
# Main
# ==================================================

def main():
    preflight_check()
    best_val_loss = (float("inf"))
    patience_counter = 0

    for epoch in range(EPOCHS):
        print(
            f"\n========== " 
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"=========="
        )
        train_loss = (train_epoch(epoch))
        val_loss = (validate_epoch())

        scheduler.step()
        print(f"\nEpoch {epoch + 1} finished")
        print(f"Train Loss: "f"{train_loss:.4f}")
        print( f"Val Loss  : "f"{val_loss:.4f}")
        print("Harmony LR:", f"{optimizer.param_groups[0]['lr']:.8f}")
        print( "Melody LR :", f"{optimizer.param_groups[1]['lr']:.8f}")
        checkpoint = (build_checkpoint( epoch, train_loss, val_loss, ))
        torch.save(checkpoint, os.path.join( GENERATION_CHECKPOINT_DIR, "generation_last.pt", ), )
        if (val_loss < best_val_loss - MIN_DELTA):
            best_val_loss = ( val_loss)
            patience_counter = 0
            torch.save(checkpoint, os.path.join( GENERATION_CHECKPOINT_DIR, "generation_best.pt",),)
            print(f"Saved best model ( Val Loss {val_loss:.4f} )")
        else:
            patience_counter += 1
            print( "No improvement: "f"{patience_counter} {PATIENCE}")
            if (patience_counter>=PATIENCE):
                print( "\nEarly stopping triggered.")
                print("Best Val Loss:", f"{best_val_loss:.4f}")
                break
    print( "\nTraining finished." )


if __name__ == "__main__":
    main()
