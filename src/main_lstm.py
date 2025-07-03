import numpy as np
import torch.multiprocessing as mp

from data.seq_dataset import seq_collate_fn, aggr_collate_fn
from plot.plot import plot_tac

mp.set_start_method("spawn", force=True)
import pickle

import hydra
import torch
import pandas as pd
from hydra.core.hydra_config import HydraConfig

from config.config import Config
from data.data_conversions import data_conversions
from data.seq_data_processing import load_data, split_data, create_dataloader, create_dataloader
from model.lstm import LSTMFeedforwardCombination
from model.mlp import MLP
import config.paths as paths
from train.linear_regression import linear_regression
from train.train_lstm import train_model
from train.eval import evaluate

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
    # df_time = df_time[:8000]
    cols_to_convert = list(cfg.training.time_feature_names)
    df_time[cols_to_convert] = df_time[cols_to_convert].astype(float)
    print("Splitting data")
    dataset_bundle = split_data(df_time, cfg.training.val_size, cfg.training.test_size, cfg.training.random_state)
    # X_train_scaled, X_val_scaled, X_test_scaled = data_splits['X_train'], data_splits['X_val'], data_splits['X_test']
    # TODO: scalen
    # correlation_analysis(X_train, y_train)
    # plot_distribution(X_train, y_train)

    output_dir = HydraConfig.get().run.dir

    seq_route_lookup = None
    aggr_route_lookup = None

    abs_accuracies_dict = {}
    relative_accuracies_dict = {}

    if cfg.compute_baseline or cfg.train_lstm:
        with open(paths.DATASETS_DIR + cfg.dataset.route_aggr + ".pkl", "rb") as f:
            aggr_route_lookup = pickle.load(f)

    if cfg.compute_baseline:
        lr_val_mae, lr_test_mae, abs_accuracies, relative_accuracies = linear_regression(cfg, dataset_bundle, aggr_route_lookup)
        print(f"Baseline MAE | val: {lr_val_mae:.2f}, test: {lr_test_mae:.2f}", flush=True)
        abs_accuracies_dict["Linear regression"] = abs_accuracies
        relative_accuracies_dict["Linear regression"] = relative_accuracies

    if cfg.train_mlp:
        model = MLP(cfg.model.input_dim, cfg.model.mlp.hidden_dims, cfg.model.output_dim)
        model.to(device)
        print("Creating dataloaders")
        train_loader = create_dataloader(cfg, dataset_bundle.train, aggr_route_lookup, cuda_num_workers=0,
                                         collate_fn=aggr_collate_fn, device=device)
        val_loader = create_dataloader(cfg, dataset_bundle.val, aggr_route_lookup, cuda_num_workers=0,
                                       collate_fn=aggr_collate_fn, device=device)
        test_loader = create_dataloader(cfg, dataset_bundle.test, aggr_route_lookup, cuda_num_workers=0,
                                        collate_fn=aggr_collate_fn, device=device)

        print("Starting training...")
        train_model(cfg, model, train_loader, val_loader, device)
        model.load_state_dict(torch.load(f"{output_dir}/{model.name}.pth"))

        mae, abs_accuracies, relative_accuracies = evaluate(cfg, model, test_loader, device)
        abs_accuracies_dict[model.name] = abs_accuracies
        relative_accuracies_dict[model.name] = relative_accuracies
        print(f"{model.name} Test MAE: {mae:.3f} ")

    if cfg.train_lstm:
        model = LSTMFeedforwardCombination(len(cfg.training.route_feature_names), cfg.model.lstm.hidden_dim, len(cfg.training.time_feature_names), cfg.model.lstm.ff_hidden_dim)
        model.to(device)

        with open(paths.DATASETS_DIR + cfg.dataset.route_seq + ".pkl", "rb") as f:
            seq_route_lookup = pickle.load(f)
        #
        # print("Creating dataloaders")
        # train_loader = create_dataloader(cfg, dataset_bundle.train, seq_route_lookup, cuda_num_workers=4, collate_fn=seq_collate_fn, device=device)
        # val_loader = create_dataloader(cfg, dataset_bundle.val, seq_route_lookup, cuda_num_workers=4, collate_fn=seq_collate_fn, device=device)
        test_loader = create_dataloader(cfg, dataset_bundle.test, seq_route_lookup, cuda_num_workers=4, collate_fn=seq_collate_fn, device=device)
        #
        # print("Starting training...")
        # train_model(cfg, model, train_loader, val_loader, device)
        model.load_state_dict(torch.load(f"model/lstm.pth", map_location=torch.device(device)))

        mae, abs_accuracies, relative_accuracies = evaluate(cfg, model, test_loader, device)
        abs_accuracies_dict[model.name] = abs_accuracies
        relative_accuracies_dict[model.name] = relative_accuracies
        print(f"{model.name} Test MAE: {mae:.3f} ")

    margins = np.arange(1, cfg.plot.margins_max, cfg.plot.step_size)
    plot_tac(margins, abs_accuracies_dict, 's', output_dir)
    margins = np.arange(1, cfg.plot.percentages_max, cfg.plot.step_size)
    plot_tac(margins, relative_accuracies_dict, 'p', output_dir)

    # with open(f"{output_dir}/results.txt", "w") as f:
    #     f.write(f"Test MAE: {mae:.3f}\n")
    #     f.write(f"Baseline MAE: {test_baseline_mae:.3f}\n")


if __name__ == "__main__":
    main()
