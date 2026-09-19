import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "traffic.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# Student Details
FULL_NAME = "Anjali Chenga"
ROLL_NUMBER = "24CE3FP45"

# Analysis Config
MORNING_PEAK_START = 5
MORNING_PEAK_END = 11
EVENING_PEAK_START = 15
EVENING_PEAK_END = 22

# Special Event Detection
Z_SCORE_THRESHOLD = 2.0

# Machine Learning Config
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
