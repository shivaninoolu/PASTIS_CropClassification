from pathlib import Path

ROOT = Path("/kaggle/input/datasets/shivaninoolu/pastis-dataset-utae2")
ROOT1 = Path("/kaggle/working/utae2")


DATASET_DIR = ROOT
S2_DIR = DATASET_DIR / "DATA_S2"
TARGET_DIR = DATASET_DIR / "ANNOTATIONS"
GEOJSON = DATASET_DIR / "metadata.geojson"

CHECKPOINT_DIR = ROOT1 / "checkpoints"
EVALUATION_DIR = ROOT1 / "evaluation"
FEATURE_DIR = ROOT1 / "feature_selection"

N_DATES = 46
N_BANDS = 10
HEIGHT = 128
WIDTH = 128

EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_WORKERS = 4

EARLY_STOPPING = True
PATIENCE = 10
MIN_DELTA = 1e-4

USE_SCHEDULER = True
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR = 1e-6

TEMPORAL_FEATURES = 32
SEED = 42

CLOUD_PERCENTILE = 97.0
CLOUD_NDVI_MAX = 0.20
CLOUD_MIN_FRACTION = 0.001
CLOUD_MAX_FRACTION = 0.35

WHOLE_DATE_CLOUD_FRACTION = 0.35

INDEX_NAMES = ["NDVI", "EVI", "LSWI", "NDRE"]

MAX_SELECTED_FEATURES = {
    "utae_selected": 6,   
    "utae_indices": 8,   
}

CORRELATION_THRESHOLD = 0.95

SCENARIOS = {
    "baseline": {
        "model": "baseline",
        "use_indices": False,
        "feature_selection": False,
    },
    "utae_selected": {
        "model": "utae",
        "use_indices": False,
        "feature_selection": True,
    },
    "utae_indices": {
        "model": "utae",
        "use_indices": True,
        "feature_selection": True,
    },
}
