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
from feature_selection.correlation_analysis import correlation_analysis
from model.mlp import MLP
import config.paths as paths
from train.train import train_model


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    df = load_data(paths.DATASETS_DIR + "/data.csv")
    df.fillna(0, inplace=True)
    X_train, X_test, y_train, y_test = split_data(df, cfg.training.test_size, cfg.training.random_state)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # correlation_analysis(X_train, y_train)

    print(X_train_scaled.shape[1])

    model = MLP(X_train_scaled.shape[1], cfg.model.mlp.hidden_dims, cfg.model.output_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = create_dataloaders(cfg, X_train, y_train, device)
    eval_loader = create_dataloaders(cfg, X_test, y_test, device)

    train_model(cfg, model, train_loader, eval_loader)


if __name__ == "__main__":
    main()
