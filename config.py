from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = ROOT_DIR / "models"

YOLO_MODEL_PATH = MODEL_DIR / "yolo_muzzle.pt"
ARCFACE_MODEL_PATH = MODEL_DIR / "arcface_muzzle.pt"


# ============================================================
# DEVICE
# ============================================================

# Your laptop does not have a CUDA GPU.
DEVICE = "cpu"


# ============================================================
# YOLO
# ============================================================

YOLO_ARCH = "yolov8n.pt"

YOLO_IMAGE_SIZE = 416
YOLO_BATCH_SIZE = 4
YOLO_EPOCHS = 30
YOLO_WORKERS = 0

YOLO_CONFIDENCE = 0.25


# ============================================================
# ARCFACE
# ============================================================

# CPU-friendly architecture
ARC_ARCH = "resnet18"

# Final biometric embedding size
ARC_EMBEDDING_DIM = 512

ARC_IMAGE_SIZE = 224

ARC_BATCH_SIZE = 16

ARC_EPOCHS = 40

ARC_WORKERS = 0

ARC_SCALE = 30.0
ARC_MARGIN = 0.5

ARC_LEARNING_RATE = 0.0003

ARC_WEIGHT_DECAY = 0.0001


# ============================================================
# DATA SPLITS
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# VERIFICATION
# ============================================================

# Cosine similarity threshold will eventually be tuned
# automatically using validation data.
DEFAULT_SIMILARITY_THRESHOLD = 0.50