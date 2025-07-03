import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
import pickle

import hydra
import torch
import pandas as pd
from hydra.core.hydra_config import HydraConfig

from config.config import Config
from data.data_conversions import data_conversions
from data.seq_data_processing import load_data, split_data, create_seq_dataloader
from model.lstm import LSTMFeedforwardCombination
from model.mlp import MLP
import config.paths as paths
from train.baseline import linear_regression
from train.train_lstm import train_model
from train.eval import test

import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(config_path=paths.CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: Config):
    if cfg.pre_data_conversions:
        data_conversions(cfg)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading time data")
    df_time = load_data(paths.DATASETS_DIR + cfg.dataset.time + '.parquet')
    print("Splitting data")
    dataset_bundle = split_data(df_time, cfg.training.val_size, cfg.training.test_size, cfg.training.random_state)
    # X_train_scaled, X_val_scaled, X_test_scaled = data_splits['X_train'], data_splits['X_val'], data_splits['X_test']
    # TODO: scalen

    seq_route_lookup = None
    aggr_route_lookup = None

    if cfg.compute_baseline or cfg.train_lstm:
        with open(paths.DATASETS_DIR + cfg.dataset.route_aggr + ".pkl", "rb") as f:
            aggr_route_lookup = pickle.load(f)


    if cfg.compute_baseline:
        lr_val_mae, lr_val_y_pred, lr_test_mae, lr_test_y_pred = linear_regression(cfg, dataset_bundle, aggr_route_lookup)
        print(f"Baseline: MAE: {lr_val_mae:.2f}")
        return

    model = LSTMFeedforwardCombination(len(cfg.training.route_feature_names), cfg.model.lstm.hidden_dim, len(cfg.training.time_feature_names), cfg.model.lstm.ff_hidden_dim)
    model.to(device)

    print(f"Loading data... ({cfg.dataset.time})", flush=True)

    with open(paths.DATASETS_DIR + cfg.dataset.route_seq + ".pkl", "rb") as f:
        seq_route_lookup = pickle.load(f)

    cols_to_convert = list(cfg.training.time_feature_names)
    df_time[cols_to_convert] = df_time[cols_to_convert].astype(float)

    # correlation_analysis(X_train, y_train)
    # plot_distribution(X_train, y_train)

    # print("Computing baseline", flush=True)
    # val_baseline_mae, val_baseline_mse, val_y_pred_baseline, test_baseline_mae, test_baseline_mse, test_y_pred_baseline = get_baseline(cfg, dataset_bundle)
    # print(f"Baseline: MAE: {val_baseline_mae:.2f} MSE: {val_baseline_mse:.2f}")

    print("Creating dataloaders")
    train_loader = create_seq_dataloader(cfg, dataset_bundle.train, seq_route_lookup, device)
    val_loader = create_seq_dataloader(cfg, dataset_bundle.val, seq_route_lookup, device)
    test_loader = create_seq_dataloader(cfg, dataset_bundle.test, seq_route_lookup, device)
    output_dir = HydraConfig.get().run.dir

    print("Starting training...")
    model, mae_list, mse_list = train_model(cfg, model, train_loader, val_loader, device)

    mse, mae = test(model, test_loader, dataset_bundle.test.y)
    print(f"Test | mse: {mse:.3f}, mae: {mae:.3f} ")
    # print(f"Baseline | mse: {test_baseline_mse:.3f}, mae: {test_baseline_mae:.3f}")
    with open(f"{output_dir}/results.txt", "w") as f:
        f.write(f"Test | mse: {mse:.3f}, mae: {mae:.3f}\n")
        # f.write(f"Baseline | mse: {test_baseline_mse:.3f}, mae: {test_baseline_mae:.3f}\n")


if __name__ == "__main__":
    main()
