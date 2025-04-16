import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

PROJECT_NAME = os.getenv("PROJECT_NAME")
ENV = os.getenv("ENV")
DATA_DIR = os.getenv("DATA_DIR")
RESULTS_DIR = os.getenv("RESULTS_DIR")
EPOCHS = os.getenv("EPOCHS")
