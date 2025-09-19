import os
from itertools import product

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from tqdm import tqdm

from config import paths
from config.config import Config
from data.data_conversions import load_route_lookup
from data.data_processing import create_dataloaders
from data.dataset_bundle import DatasetBundle
from model.lstm import LSTMFeedforwardCombination
from model.mlp import MLP
from train.train import train_model

def lstm_grid_search(cfg: Config):
    print("Starting LSTM GridSearch")
    gs_lstm_hidden_dim = [16, 32, 64]
    gs_num_lstm_layers = [1]
    gs_bidirectional = [True, False]
    gs_dropout = [0.0, 0.2]
    gs_ff_hidden_dims = [[16], [32, 16]]
    gs_learning_rate = [1e-4, 3e-4, 1e-3]
    gs_weight_decay = [0.0, 1e-4]
    iterations = (len(gs_lstm_hidden_dim) * len(gs_num_lstm_layers) * len(gs_bidirectional) * len(gs_dropout) *
                  len(gs_ff_hidden_dims) * len(gs_learning_rate) * len(gs_weight_decay))

    output_dir = HydraConfig.get().run.dir
    config_path = output_dir + "/configs"
    os.makedirs(config_path)
    losses_path = output_dir + "/losses"
    os.makedirs(losses_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4 if device.type == 'cuda' else 0
    dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR)
    seq_route_lookup = load_route_lookup(paths.DATASETS_DIR + cfg.dataset.route_seq)
    train_loader, val_loader, test_loader = create_dataloaders(cfg, dataset_bundle, seq_route_lookup,
                                                               is_route_sequence=True, num_workers=num_workers)
    best_mae = np.inf
    best_idx = -1

    results_path = output_dir + "/results.csv"
    df_results = pd.DataFrame({'idx': pd.Series(dtype=int), 'score': pd.Series(dtype=float)})
    df_results.to_csv(results_path, index=False)

    for idx, (l_hidden, l_layers, bi, do, ff_hidden, lr, wd) in tqdm(enumerate(product(gs_lstm_hidden_dim, gs_num_lstm_layers,gs_bidirectional, gs_dropout, gs_ff_hidden_dims, gs_learning_rate, gs_weight_decay)), total=iterations):
        cfg.model.lstm.lstm_hidden_dim = l_hidden
        cfg.model.lstm.num_lstm_layers = l_layers
        cfg.model.lstm.bidirectional = bi
        cfg.model.lstm.dropout = do
        cfg.model.lstm.ff_hidden_dims = ff_hidden
        cfg.training.optimizer_lstm.learning_rate = lr
        cfg.training.optimizer_lstm.weight_decay = wd
        model = LSTMFeedforwardCombination(cfg)
        model.to(device)
        train_losses, val_losses, best_id_targets, val_mae = train_model(cfg, model, train_loader, val_loader,
                                                                         cfg.training.optimizer_mlp, device, verbose=False)
        OmegaConf.save(cfg, config_path + f"/config{idx}.yaml", resolve=True)
        np.save(losses_path + f"/train_{idx}.npy", train_losses)
        np.save(losses_path + f"/val_{idx}.npy", val_losses)
        if val_mae < best_mae:
            best_mae = val_mae
            best_idx = idx

        df = pd.DataFrame([[idx, val_mae]], columns=["idx", "score"])
        df.to_csv(results_path, mode="a", header=False, index=False)

    print(f"Best cfg: {best_idx}, MAE: {best_mae}")

def mlp_grid_search(cfg: Config):
    print("Starting MLP GridSearch")
    gs_dropout = [0.0, 0.1, 0.2]
    gs_hidden_dims = [[64, 32], [128, 64], [128, 64, 32]]
    gs_learning_rate = [1e-3, 5e-3, 1e-4]
    gs_weight_decay = [0.0, 1e-5, 1e-4]

    # [6, 7, 8, 15, 16, 17, 24, 25, 26, 27, 29, 33, 34, 35, 42, 43, 44, 51, 52, 53, 54, 57, 60, 61, 62, 69, 70, 71, 78, 79, 80]
    # idx_not_finished = [6, 7, 8, 15, 16, 17, 24, 25, 26]
    idx_not_finished = [27, 29, 33, 34, 35, 42, 43, 44]
    # idx_not_finished = [51, 52, 53, 54, 57, 60, 61]
    # idx_not_finished =  [62, 69, 70, 71, 78, 79, 80]

    iterations = len(gs_dropout) * len(gs_hidden_dims) * len(gs_learning_rate) * len(gs_weight_decay)

    output_dir = HydraConfig.get().run.dir
    config_path = output_dir + "/configs"
    os.makedirs(config_path)
    losses_path = output_dir + "/losses"
    os.makedirs(losses_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4 if device.type == 'cuda' else 0
    dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR)
    aggr_route_lookup = load_route_lookup(paths.DATASETS_DIR + cfg.dataset.route_aggr)
    train_loader, val_loader, test_loader = create_dataloaders(cfg, dataset_bundle, aggr_route_lookup,
                                                               is_route_sequence=False, num_workers=num_workers)
    best_mae = np.inf
    best_idx = -1

    results_path = output_dir + "/results.csv"
    df_results = pd.DataFrame({'idx': pd.Series(dtype=int), 'score': pd.Series(dtype=float)})
    df_results.to_csv(results_path, index=False)
    input_dim = dataset_bundle.train.x.shape[1] - 2 + next(iter(aggr_route_lookup.values())).shape[1]
    for idx, (dropout, hidden, lr, wd) in tqdm(enumerate(product(gs_dropout, gs_hidden_dims, gs_learning_rate, gs_weight_decay)), total=iterations):
        # idx = iterations - 1 - idx
        if idx not in idx_not_finished:
            continue
        cfg.model.mlp.dropout = dropout
        cfg.model.mlp.hidden_dims = hidden
        cfg.training.optimizer_mlp.learning_rate = lr
        cfg.training.optimizer_mlp.weight_decay = wd

        model = MLP(cfg, input_dim)
        model.to(device)
        train_losses, val_losses, best_id_targets, val_mae = train_model(cfg, model, train_loader, val_loader,
                                                                         cfg.training.optimizer_mlp, device, verbose=False)
        OmegaConf.save(cfg, config_path + f"/config{idx}.yaml", resolve=True)
        np.save(losses_path + f"/train_{idx}.npy", train_losses)
        np.save(losses_path + f"/val_{idx}.npy", val_losses)
        if val_mae < best_mae:
            best_mae = val_mae
            best_idx = idx

        df = pd.DataFrame([[idx, val_mae]], columns=["idx", "score"])
        df.to_csv(results_path, mode="a", header=False, index=False)

    print(f"Best cfg: {best_idx}, MAE: {best_mae}")

@hydra.main(config_path=paths.CONFIG_DIR, config_name="config_gs", version_base=None)
def main(cfg: Config):
    if cfg.train_mlp:
        mlp_grid_search(cfg)
    elif cfg.train_lstm:
        lstm_grid_search(cfg)

if __name__ == "__main__":
    main()
