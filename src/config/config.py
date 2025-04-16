from dataclasses import dataclass

from config.env import PROJECT_NAME, ENV, DATA_DIR, EPOCHS, RESULTS_DIR
from config.args import get_parsed_args
from pathlib import Path

@dataclass
class Config:
    project_name: str
    env: str
    data_dir: Path
    results_dir: Path
    epochs: int

def load_config():
    args = get_parsed_args()

    config = Config(
        project_name=args.project_name or PROJECT_NAME,
        env=ENV,
        data_dir=Path(args.data_dir or DATA_DIR),
        results_dir=Path(RESULTS_DIR),
        epochs=args.epochs or EPOCHS
    )

    return config
