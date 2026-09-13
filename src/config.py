import os

PROJECT_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

# ==========================
# Checkpoint
# ==========================

CHECKPOINT_DIR = os.path.join(BASE_DIR,"Checkpoint")
HARMONY_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT,"Checkpoints","Harmony")
MELODY_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT,"Checkpoints","Melody")
GENERATION_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT,"Checkpoints","Generation")
# ==========================
# Output
# ==========================

OUTPUT_DIR = os.path.join(BASE_DIR,"Output")