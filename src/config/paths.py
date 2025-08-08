import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS_DIR = os.path.join(ROOT_DIR, "datasets/")
DATASET_BUNDLE_DIR = os.path.join(DATASETS_DIR, "splits/")
RESULTS_DIR = os.path.join(ROOT_DIR, "results/")
CONFIG_DIR = os.path.join(ROOT_DIR, "config/")
HYDRA_OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
