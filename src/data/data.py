import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from config.config import Config


def load_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def split_data(df: pd.DataFrame, test_size, random_state) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df["recorded_elapsed_time"]
    X = df.drop(columns=["recorded_elapsed_time"])

    unique_ids = X['stop_to_stop_id'].unique()

    train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)

    train_mask = X['stop_to_stop_id'].isin(train_ids)
    test_mask = X['stop_to_stop_id'].isin(test_ids)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    return X_train, X_test, y_train, y_test


def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_scaled, X_test_scaled

def create_dataloaders(cfg: Config, X: pd.DataFrame, y: pd.DataFrame, device) -> DataLoader:
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataLoader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)
    return dataLoader