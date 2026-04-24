"""Filesystem paths used by the data and training pipelines."""
import os
import socket

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Datasets live in the workspace by default but on the HPC scratch FS in production.
if socket.gethostname().startswith("linux"):
    DATASETS_DIR = os.path.join(ROOT_DIR, "datasets/")
else:
    DATASETS_DIR = "/scratch/s3799174/datasets/"

DATASET_BUNDLE_DIR = os.path.join(DATASETS_DIR, "splits/")
RESULTS_DIR = os.path.join(ROOT_DIR, "results/")
CONFIG_DIR = os.path.join(ROOT_DIR, "config/")
HYDRA_OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
