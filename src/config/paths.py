import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS_DIR = os.path.join(ROOT_DIR, "datasets/")
RESULTS_DIR = os.path.join(ROOT_DIR, "results/")
CONFIG_DIR = os.path.join(ROOT_DIR, "src/config/")
HYDRA_OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
