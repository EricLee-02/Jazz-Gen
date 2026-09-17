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
MELODY_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR,"Melody")
MELODY_CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR,"Melody","melody_best.pt")
GENERATION_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR,"Generation","generation_best.pt")
# ==========================
# Output
# ==========================

OUTPUT_DIR = os.path.join(BASE_DIR,"Output")

DEVICE = "cuda"

# Melody Generation Config

TOKENS_PER_BAR = 55
BARS_32 = 32
BARS_48 = 48
BARS_64 = 64

GENERATE_32_BAR_LENGTH = BARS_32 * TOKENS_PER_BAR
GENERATE_48_BAR_LENGTH = BARS_48 * TOKENS_PER_BAR
GENERATE_64_BAR_LENGTH = BARS_64 * TOKENS_PER_BAR

TEMPERATURE = 0.85
TOP_K = 40


#Melody Decoder
MELODY_MAX_SEQ_LEN = 4096
STRIDE = 512
MELODY_D_MODEL = 512
MELODY_HEADS = 8
MELODY_LAYERS = 8
TRAIN_RATIO = 0.9
DROPOUT=0.1
EPOCHS = 50
BATCH_SIZE = 8
LR = 3e-4
PATIENCE = 8
WEIGHT_DECAY = 0.01

#Harmony Encoder
CHORD_VOCAB_SIZE=1064
DURATION_VOCAB_SIZE=10
BEAT_VOCAB_SIZE=20
SECTION_VOCAB_SIZE=20
TIME_VOCAB_SIZE=10
D_MODEL=512
NUM_HEADS=8
NUM_LAYERS=6
COUNTER = 0
BEST_LOSS = 999
