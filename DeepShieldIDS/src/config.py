
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"

CIC_PATH = RAW_DIR / "cicids2017.csv"
UNSW_PATH = RAW_DIR / "unsw_nb15.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.10
VAL_SIZE = 0.10

BINARY_MODE = True
TARGET_ATTACK_LABELS = {"BENIGN", "NORMAL", "Normal", "normal", "0", 0}

IRRELEVANT_COLUMN_HINTS = [
    "flow id", "flow_id", "timestamp", "time", "src ip", "source ip", "dst ip", "destination ip",
    "source", "destination", "srcip", "dstip", "id", "attack_cat"
]
LABEL_CANDIDATES = ["Label", "label", "class", "Class", "attack", "Attack", "target", "Target"]

DEFAULT_MODEL_PARAMS = {
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 40,
    "patience": 8,
    "cnn_filters": 64,
    "kernel_size": 3,
    "lstm_units": 128,
    "dropout": 0.3,
    "dense_units": 128,
    "l2_reg": 1e-3,
}
