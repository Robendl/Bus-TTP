"""Process-level setup shared by every entry-point script."""
import os

import torch.multiprocessing as mp


def setup_environment() -> None:
    mp.set_start_method("spawn", force=True)
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
