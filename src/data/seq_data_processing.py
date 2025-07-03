import torch
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from config.config import Config
from data.dataset_bundle import DatasetBundle, DatasetSplit
from data.seq_dataset import SequenceDataset


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

def create_dataloader(cfg: Config, dataset_split: DatasetSplit, route_lookup, collate_fn, cuda_num_workers, device) -> DataLoader:
    dataset = SequenceDataset(dataset_split, route_lookup, cfg.training.time_feature_names, cfg.training.route_feature_names)

    if device.type == 'cuda':
        num_workers = cuda_num_workers
    else:
        num_workers = 0

    data_loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    return data_loader
