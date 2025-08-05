import torch
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config.config import Config
from data.dataset_bundle import DatasetBundle, DatasetSplit
from data.mapping_dataset import MappingDataset

def split_data(cfg: Config, df: pd.DataFrame) -> DatasetBundle:
    val_size = cfg.training.val_size
    test_size = cfg.training.test_size
    random_state = cfg.training.random_state
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

def scale_time_features(cfg: Config, dataset_bundle):
    time_cols = list(cfg.dataset.scaling_time_features)
    scaler = StandardScaler()
    scaler.fit(dataset_bundle.train.x[time_cols])


    for split in [dataset_bundle.train, dataset_bundle.val, dataset_bundle.test]:
        split.x[time_cols] = pd.DataFrame(scaler.transform(
            split.x[time_cols]),
            columns=time_cols,
            index=split.x.index
        )

    return dataset_bundle

def scale_route_lookup(cfg:Config, df: pd.DataFrame, train_hashes: set):
    scaling_features = list(cfg.dataset.scaling_route_features)
    train_df = df[df["route_seq_hash"].isin(train_hashes)]
    stacked_train_data = train_df[scaling_features].values.astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(stacked_train_data)
    df_scaled = df.copy()
    df_scaled[scaling_features] = scaler.transform(df_scaled[scaling_features].values.astype(np.float32))

    return df_scaled


def create_dataloader(cfg: Config, dataset_split: DatasetSplit, route_lookup, route_feature_indices, collate_fn, num_workers) -> DataLoader:
    dataset = MappingDataset(dataset_split, route_lookup, cfg.dataset.time_feature_names, route_feature_indices)
    data_loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    return data_loader

def create_dataloaders(cfg: Config, dataset_bundle: DatasetBundle, route_lookup, collate_fn, num_workers) -> Tuple[DataLoader, DataLoader, DataLoader]:
    route_feature_indices = [cfg.dataset.route_feature_names_full.index(name) for name in cfg.dataset.route_feature_names]
    train_loader = create_dataloader(cfg, dataset_bundle.train, route_lookup, route_feature_indices, collate_fn, num_workers)
    val_loader = create_dataloader(cfg, dataset_bundle.val, route_lookup, route_feature_indices, collate_fn, num_workers)
    test_loader = create_dataloader(cfg, dataset_bundle.test, route_lookup, route_feature_indices, collate_fn, num_workers)
    return train_loader, val_loader, test_loader
