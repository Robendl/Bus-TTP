import os
import json
import torch
# import yaml
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

from config.args import get_parsed_args
from pathlib import Path

@dataclass
class Config_old:
    project_name: str
    env: str
    data_dir: Path
    results_dir: Path
    epochs: int
    mlp_hidden_dims: List[int]
    device: torch.device

def load_config():
    args = get_parsed_args()
    load_dotenv()
    # with open("")

    config = Config_old(
        project_name=args.project_name or os.getenv("PROJECT_NAME"),
        env=os.getenv("ENV"),
        data_dir=Path(args.data_dir or os.getenv("DATA_DIR")),
        results_dir=Path(os.getenv("RESULTS_DIR")),
        epochs=args.epochs or os.getenv("EPOCHS"),
        mlp_hidden_dims=json.loads(os.getenv("MLP_HIDDEN_DIMS")),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    return config
