import os
from pathlib import Path

import hydra
import numpy as np
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from torch.utils.data import TensorDataset, DataLoader

from config.config import Config
from data.data import load_data, split_data, scale_data, create_dataloaders
from data.plot_distribution import plot_distribution
from feature_selection.correlation_analysis import correlation_analysis
from model.mlp import MLP
import config.paths as paths
from plot.plot import plot_results
from train.baseline import get_baseline
from train.train import train_model


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    print("Loading data...")
    df = load_data(paths.DATASETS_DIR + "/training_13juni_20p.csv")
    df.drop(columns=["recordeddeparturetime"], inplace=True)
    print("Filling 0's")
    df.info(memory_usage='deep')
    df.fillna(0, inplace=True)
    df.info(memory_usage='deep')
    print("Splitting data")
    X_train, X_test, y_train, y_test = split_data(df, cfg.training.test_size, cfg.training.random_state)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # correlation_analysis(X_train, y_train)

    # plot_distribution(X_train, y_train)
    print("Scaling data")
    X_train_scaled.drop(columns=["stop_to_stop_id"], inplace=True)
    X_test_scaled.drop(columns=["stop_to_stop_id"], inplace=True)

    print("Computing baseline")
    baseline_mae, baseline_mse = get_baseline(X_train_scaled, y_train, X_test_scaled, y_test)
    print(f"Baseline: MAE: {baseline_mae} MSE: {baseline_mse}")

    model = MLP(X_train_scaled.shape[1], cfg.model.mlp.hidden_dims, cfg.model.output_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = create_dataloaders(cfg, X_train_scaled, y_train, device)
    eval_loader = create_dataloaders(cfg, X_test_scaled, y_test, device)

    print("Starting training...")
    model, mae_list, mse_list = train_model(cfg, model, train_loader, eval_loader)
    plot_results(mae_list, mse_list, baseline_mae, baseline_mse)


if __name__ == "__main__":
    main()
