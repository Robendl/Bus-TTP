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


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config_gs", version_base=None)
def main(cfg: Config):
    dataset_bundle = DatasetBundle.load(paths.DATASET_BUNDLE_DIR, cfg)
    print("loaded dataset bundle", flush=True)
    seq_route_lookup = load_route_lookup(cfg,
        paths.DATASETS_DIR + cfg.dataset.route_seq)
    print("loaded route lookup", flush=True)
    lstm_input_dim = next(iter(seq_route_lookup.values())).shape[1]
    ff_input_dim = dataset_bundle.train.x.shape[1] - 2
    model = LSTMFeedforwardCombination(cfg, lstm_input_dim, ff_input_dim)
    model.load_state_dict(torch.load("outputs/2025-10-01/16-07-36/LSTM.pth"))
    print("loaded model", flush=True)
    train_loader, val_loader, test_loader = create_dataloaders(cfg, dataset_bundle, seq_route_lookup,
                                                               is_route_sequence=True, num_workers=0)
    _, (time_features, padded_routes, lengths), _ = next(iter(train_loader))
    dummy_input = (time_features[0:1], padded_routes[0:1], lengths[0:1])
    print("start export", flush=True)
    torch.onnx.export(
        model,
        dummy_input,
        f"{paths.RESULTS_DIR}/onnx/LSTM.onnx",
        input_names=["time_features", "padded_routes", "lengths"],
        output_names=["output"],
        dynamic_axes={
            "time_features": {0: "batch_size"},
            "padded_routes": {0: "batch_size", 1: "seq_len"},
            "lengths": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

if __name__ == "__main__":
    main()
