import os

# ==========================
# Google Drive Root
# ==========================

BASE_DIR = "/content/drive/MyDrive/JazzGen_Data"
# ==========================
# Harmony Encoder
# ==========================
# iReal Pro harmony token/json
HARMONY_TOKEN_DIR = os.path.join(BASE_DIR,"Ireal_Harmony_Token")
# harmony vocabulary
HARMONY_VOCAB_DIR = os.path.join(BASE_DIR,"Harmony_Vocabulary")
# ==========================
# Melody Decoder
# ==========================

# WJazzD / solo token
MELODY_TOKEN_DIR = os.path.join(BASE_DIR,"Melody_Token")
# Melody vocabulary
MELODY_VOCAB_DIR = os.path.join(BASE_DIR,"Melody_Vocabulary")
MELODY_VOCAB_FILE = os.path.join(BASE_DIR,"Melody_Vocabulary","vocabulary.json")


# ==========================
# Checkpoint
# ==========================

CHECKPOINT_DIR = os.path.join(BASE_DIR,"Checkpoints")
HARMONY_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR,"Harmony","harmony_best.pt")
MELODY_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR,"Melody","melody_best.pt")
GENERATION_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR,"Generation")
# ==========================
# Output
# ==========================

OUTPUT_DIR = os.path.join(BASE_DIR,"Output")