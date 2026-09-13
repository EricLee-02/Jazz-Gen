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

# ==========================
# Checkpoint
# ==========================

CHECKPOINT_DIR = os.path.join(BASE_DIR,"Checkpoint")
HARMONY_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR,"Harmony")
MELODY_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR,"Melody")
GENERATION_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR,"Generation")
# ==========================
# Output
# ==========================

OUTPUT_DIR = os.path.join(BASE_DIR,"Output")