import os
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import config.paths as paths

from config.config import Config
from data.dataset_bundle import DatasetBundle, DatasetSplit
from data.seq_dataset import SequenceDataset, CollateFn


def load_data(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df

def split_data(df: pd.DataFrame, val_size, test_size, random_state) -> DatasetBundle:
    y = df["recorded_elapsed_time"]
    X = df.drop(columns=["recorded_elapsed_time"])

    unique_ids = X['stop_to_stop_id'].unique()
    rng = np.random.default_rng(seed=random_state)
    shuffled_ids = rng.permutation(unique_ids)

    n_total = len(shuffled_ids)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)

    test_ids = shuffled_ids[:n_test]
    val_ids = shuffled_ids[n_test:n_test + n_val]
    train_ids = shuffled_ids[n_test + n_val:]

    # Create masks
    train_mask = X['stop_to_stop_id'].isin(train_ids)
    val_mask = X['stop_to_stop_id'].isin(val_ids)
    test_mask = X['stop_to_stop_id'].isin(test_ids)

    X = X.drop(columns=["stop_to_stop_id"])

    # Final splits
    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

    return DatasetBundle(
        train=DatasetSplit(X_train, y_train),
        val=DatasetSplit(X_val, y_val),
        test=DatasetSplit(X_test, y_test)
    )

def scale_data(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_scaled, X_val_scaled, X_test_scaled

def create_dataloader(cfg: Config, X: pd.DataFrame, y: pd.DataFrame, device) -> DataLoader:
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataLoader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)
    return dataLoader

def prepare_data(cfg: Config, device):
    print(f"Loading data... ({"seq"})", flush=True)
    df_route = load_data(paths.DATASETS_DIR + 'seq_dataset_rf.csv')
    df_time = load_data(paths.DATASETS_DIR + 'seq_dataset_tf.csv')
    # print("Filling 0's")
    # df_route.fillna(0, inplace=True)
    print("Splitting data")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_time, cfg.training.val_size, cfg.training.test_size,
                                                                cfg.training.random_state)
    X_train_scaled, X_val_scaled, X_test_scaled = X_train, X_val, X_test
    # TODO: scalen

    # correlation_analysis(X_train, y_train)
    # plot_distribution(X_train, y_train)

    X_train_scaled.drop(columns=["stop_to_stop_id"], inplace=True)
    X_val_scaled.drop(columns=["stop_to_stop_id"], inplace=True)
    X_test_scaled.drop(columns=["stop_to_stop_id"], inplace=True)

    # print("Computing baseline", flush=True)
    # val_baseline_mae, val_baseline_mse, val_y_pred_baseline, test_baseline_mae, test_baseline_mse, test_y_pred_baseline = get_baseline(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
    # print(f"Baseline: MAE: {val_baseline_mae:.2f} MSE: {val_baseline_mse:.2f}")

    train_loader = create_seq_dataloader(cfg, X_train_scaled, y_train, device)
    val_loader = create_seq_dataloader(cfg, X_val_scaled, y_val, device)
    test_loader = create_seq_dataloader(cfg, X_test_scaled, y_test, device)
    return train_loader, val_loader, test_loader

def create_seq_dataloader(cfg: Config, dataset_split: DatasetSplit, route_lookup, device) -> DataLoader:
    dataset = SequenceDataset(dataset_split, route_lookup, cfg.training.time_feature_names, cfg.training.route_feature_names, device)
    collate_fn = CollateFn(device)
    if device.type == 'cuda':
        num_workers = 3
    else:
        num_workers = 0
    dataLoader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    return dataLoader
