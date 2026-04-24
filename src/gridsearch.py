"""Hydra-driven grid search over MLP and LSTM hyperparameters.

Each combination is logged as ``configN.yaml`` together with its train / val
loss curves and final validation MAE. Resulting CSV is written to
``<run_dir>/results.csv``.
"""
import os
from itertools import product
from pathlib import Path
from typing import Callable, Iterable

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from tqdm import tqdm

from config import paths
from config.config import Config
from data.build_dataset import load_route_lookup
from data.data_processing import create_dataloaders
from data.dataset_bundle import DatasetBundle
from model.lstm import LSTMFeedforwardCombination
from model.mlp import MLP
from runtime import setup_environment
from train.train import train_model

setup_environment()


def _prepare_output_dirs() -> tuple[Path, Path, Path]:
    output_dir = Path(HydraConfig.get().run.dir)
    config_path = output_dir / "configs"
    losses_path = output_dir / "losses"
    config_path.mkdir(parents=True, exist_ok=True)
    losses_path.mkdir(parents=True, exist_ok=True)
    return output_dir, config_path, losses_path


def _run_grid(
    cfg: Config,
    grid: Iterable[tuple],
    apply_combo: Callable[[Config, tuple], None],
    build_model: Callable[[Config], torch.nn.Module],
    optimizer_cfg,
    train_loader,
    val_loader,
    device: torch.device,
    total_iterations: int,
) -> None:
    output_dir, config_path, losses_path = _prepare_output_dirs()
    results_path = output_dir / "results.csv"
    pd.DataFrame({"idx": pd.Series(dtype=int), "score": pd.Series(dtype=float)}).to_csv(
        results_path, index=False
    )

    best_mae = np.inf
    best_idx = -1

    for idx, combo in tqdm(enumerate(grid), total=total_iterations):
        apply_combo(cfg, combo)
        model = build_model(cfg).to(device)
        train_losses, val_losses, _, val_mae, _, _ = train_model(
            cfg, model, train_loader, val_loader, optimizer_cfg, device, verbose=False
        )

        OmegaConf.save(cfg, config_path / f"config{idx}.yaml", resolve=True)
        np.save(losses_path / f"train_{idx}.npy", train_losses)
        np.save(losses_path / f"val_{idx}.npy", val_losses)

        if val_mae < best_mae:
            best_mae = val_mae
            best_idx = idx

        pd.DataFrame([[idx, val_mae]], columns=["idx", "score"]).to_csv(
            results_path, mode="a", header=False, index=False
        )

    print(f"Best config: {best_idx} | MAE: {best_mae:.3f}")


def lstm_grid_search(cfg: Config) -> None:
    print("Starting LSTM grid search")
    grid = list(product(
        [16, 32, 64],          # lstm_hidden_dim
        [1],                   # num_lstm_layers
        [True, False],         # bidirectional
        [0.0, 0.2],            # dropout
        [[16], [32, 16]],      # ff_hidden_dims
        [1e-4, 3e-4, 1e-3],    # learning_rate
        [0.0, 1e-4],           # weight_decay
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4 if device.type == "cuda" else 0

    dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR, cfg)
    seq_route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_seq)
    train_loader, val_loader, _ = create_dataloaders(
        cfg, dataset_bundle, seq_route_lookup, is_route_sequence=True, num_workers=num_workers
    )

    def apply_combo(cfg: Config, combo: tuple) -> None:
        (cfg.model.lstm.lstm_hidden_dim, cfg.model.lstm.num_lstm_layers,
         cfg.model.lstm.bidirectional, cfg.model.lstm.dropout, cfg.model.lstm.ff_hidden_dims,
         cfg.training.optimizer_lstm.learning_rate, cfg.training.optimizer_lstm.weight_decay) = combo

    def build_model(cfg: Config) -> torch.nn.Module:
        lstm_input_dim = next(iter(seq_route_lookup.values())).shape[1]
        ff_input_dim = dataset_bundle.train.x.shape[1] - 2
        return LSTMFeedforwardCombination(cfg, lstm_input_dim, ff_input_dim)

    _run_grid(
        cfg, grid, apply_combo, build_model,
        cfg.training.optimizer_lstm, train_loader, val_loader, device, len(grid),
    )


def mlp_grid_search(cfg: Config) -> None:
    print("Starting MLP grid search")
    grid = list(product(
        [0.0, 0.1, 0.2],                        # dropout
        [[64, 32], [128, 64], [128, 64, 32]],   # hidden_dims
        [1e-3, 5e-3, 1e-4],                     # learning_rate
        [0.0, 1e-5, 1e-4],                      # weight_decay
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4 if device.type == "cuda" else 0

    dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR, cfg)
    aggr_route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_aggr)
    train_loader, val_loader, _ = create_dataloaders(
        cfg, dataset_bundle, aggr_route_lookup, is_route_sequence=False, num_workers=num_workers
    )

    def apply_combo(cfg: Config, combo: tuple) -> None:
        (cfg.model.mlp.dropout, cfg.model.mlp.hidden_dims,
         cfg.training.optimizer_mlp.learning_rate, cfg.training.optimizer_mlp.weight_decay) = combo

    def build_model(cfg: Config) -> torch.nn.Module:
        input_dim = dataset_bundle.train.x.shape[1] - 2 + next(iter(aggr_route_lookup.values())).shape[1]
        return MLP(cfg, input_dim)

    _run_grid(
        cfg, grid, apply_combo, build_model,
        cfg.training.optimizer_mlp, train_loader, val_loader, device, len(grid),
    )


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config_gs", version_base=None)
def main(cfg: Config):
    if cfg.train_mlp:
        mlp_grid_search(cfg)
    elif cfg.train_lstm:
        lstm_grid_search(cfg)
    else:
        print("Neither train_mlp nor train_lstm enabled in config")


if __name__ == "__main__":
    main()
