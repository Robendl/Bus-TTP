import pandas as pd
import torch.multiprocessing as mp

from plot.analysis import validation_analysis, get_od_results, bootstrap_ci, paired_significance_test, residual_plots
from train.xgboost import train_xgb

mp.set_start_method("spawn", force=True)
import hydra
import torch
import numpy as np
from hydra.core.hydra_config import HydraConfig

from data.dataset_bundle import DatasetBundle
from plot.plot import bootstrap_tac_per_model
from config.config import Config
from data.data_conversions import data_conversions, load_route_lookup
from data.data_processing import create_dataloaders
from model.lstm import LSTMFeedforwardCombination
from model.mlp import MLP
import config.paths as paths
from train.linear_regression import linear_regression
from train.train import train_model
from train.eval import evaluate

import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["HYDRA_FULL_ERROR"] = "1"

def run_training(cfg, model, route_lookup, dataset_bundle, num_workers, cfg_optim, device, output_dir, is_route_sequence):
    train_loader, val_loader, test_loader = create_dataloaders(cfg, dataset_bundle, route_lookup,
                                                               is_route_sequence, num_workers)
    train_model(cfg, model, train_loader, val_loader, cfg_optim, device)

@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    print(f"Using dataset: {cfg.dataset.time}")
    if cfg.pre_data_conversions:
        data_conversions(cfg)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading time data")
    dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR, cfg)
    print(dataset_bundle.train.x.shape)
    result_strings = []
    output_dir = HydraConfig.get().run.dir
    num_workers = 4 if device.type == 'cuda' else 0
    print(f"num workers: {num_workers}")

    aggr_route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_aggr)

    if cfg.train_mlp:
        input_dim = dataset_bundle.train.x.shape[1] - 3 + next(iter(aggr_route_lookup.values())).shape[1]
        print(f"MLP input dim: {input_dim}")
        model = MLP(cfg, input_dim)
        model.to(device)
        run_training(cfg, model, aggr_route_lookup, dataset_bundle, num_workers, cfg.training.optimizer_mlp, device,
                     output_dir, is_route_sequence=False)

    if cfg.train_lstm:
        print("Loading sequence route lookup", flush=True)
        seq_route_lookup = load_route_lookup(cfg, paths.DATASETS_DIR + cfg.dataset.route_seq)
        lstm_input_dim = next(iter(seq_route_lookup.values())).shape[1]
        ff_input_dim = dataset_bundle.train.x.shape[1] - 3
        model = LSTMFeedforwardCombination(cfg, lstm_input_dim, ff_input_dim)
        model.to(device)
        run_training(cfg, model, seq_route_lookup, dataset_bundle, num_workers, cfg.training.optimizer_lstm, device,
                     output_dir, is_route_sequence=True)

if __name__ == "__main__":
    main()
