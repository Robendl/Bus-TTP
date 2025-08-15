import os
from itertools import product

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from config import paths
from config.config import Config
from data.data_conversions import load_route_lookup
from data.data_processing import create_dataloaders
from data.dataset_bundle import DatasetBundle
from model.mlp import MLP
from train.train import train_model


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config_gs", version_base=None)
def mlp_grid_search(cfg: Config):
    gs_dropout = [0.0, 0.1, 0.2]
    gs_hidden_dims = [[64, 32], [128, 64], [128, 64, 32]]
    gs_learning_rate = [1e-3, 5e-3, 1e-4]
    gs_weight_decay = [0.0, 1e-5, 1e-4]

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
    for idx, (dropout, hidden, lr, wd) in enumerate(product(gs_dropout, gs_hidden_dims, gs_learning_rate, gs_weight_decay)):
        if idx > 0:
            train_loader, val_loader, test_loader = create_dataloaders(cfg, dataset_bundle, aggr_route_lookup,
                                                                       is_route_sequence=False, num_workers=0)
        cfg.model.mlp.dropout = dropout
        cfg.model.mlp.hidden_dims = hidden
        cfg.training.optimizer_mlp.learning_rate = lr
        cfg.training.optimizer_mlp.weight_decay = wd
        model = MLP(cfg)
        train_losses, val_losses, best_id_targets, val_mae = train_model(cfg, model, train_loader, val_loader,
                                                                         cfg.training.optimizer_mlp, device, verbose=True)
        OmegaConf.save(cfg, config_path + f"/config{idx}.yaml", resolve=True)
        np.save(losses_path + f"/train_{idx}.npy", train_losses)
        np.save(losses_path + f"/val_{idx}.npy", val_losses)
        print(f"{idx}: Val MAE: {val_mae}")
        if val_mae < best_mae:
            best_mae = val_mae
            best_idx = idx

        df = pd.DataFrame([[idx, val_mae]], columns=["idx", "score"])
        df.to_csv(results_path, mode="a", header=False, index=False)

    print(f"Best cfg: {best_idx}, MAE: {best_mae}")



if __name__ == "__main__":
    mlp_grid_search()