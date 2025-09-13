import os

import torch
import pandas as pd
import numpy as np
from typing import Tuple, Dict

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from config import paths
from config.config import Config
from data.dataset_bundle import DatasetBundle, DatasetSplit
from data.mapping_dataset import MappingDataset, seq_collate_fn, aggr_collate_fn
from data.route_based_dataset import route_based_aggr_collate_fn, RouteBasedDataset, route_based_seq_collate_fn


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

    if not cfg.dataset.multi_run:
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

def save_sklearn(sklearn_obj, n_features, filename):
    initial_type = [('input', FloatTensorType([None, n_features]))]
    onnx_preprocess = convert_sklearn(sklearn_obj, initial_types=initial_type)
    onnx_dir = f"{paths.RESULTS_DIR}/onnx"
    os.makedirs(onnx_dir, exist_ok=True)
    with open(f"{onnx_dir}/{filename}", "wb") as f:
        f.write(onnx_preprocess.SerializeToString())

def scale_time_features(cfg: Config, dataset_bundle):
    time_cols = list(cfg.dataset.time_feature_names)
    scaling_time_cols = list(cfg.dataset.scaling_time_features)

    scaling_idx = [time_cols.index(c) for c in scaling_time_cols]
    passthrough_idx = [i for i, c in enumerate(time_cols) if c not in scaling_time_cols]

    scaling_ct = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), scaling_idx),
            ("passthrough", "passthrough", passthrough_idx)
        ]
    )

    pipeline_steps = []
    pipeline_steps.append(("scaling", scaling_ct))
    if cfg.dataset.pca:
        pipeline_steps.append(("pca", PCA(n_components=0.95)))

    preprocessing_pipe = Pipeline(pipeline_steps)

    preprocessing_pipe.fit(dataset_bundle.train.x[time_cols].values)

    if not cfg.dataset.multi_run:
        splits = [dataset_bundle.train, dataset_bundle.val, dataset_bundle.test]
    else:
        splits = [dataset_bundle.train, dataset_bundle.test]

    for split in splits:
        features = split.x[time_cols].values  # numpy array
        ids = split.x.drop(columns=time_cols)

        processed = preprocessing_pipe.transform(features)

        if cfg.dataset.pca:
            new_cols = [f"pca_time_{i}" for i in range(processed.shape[1])]
        else:
            new_cols = scaling_time_cols + [c for c in time_cols if c not in scaling_time_cols]

        processed_df = pd.DataFrame(processed, index=ids.index, columns=new_cols)
        split.x = pd.concat([ids, processed_df], axis=1)

    save_sklearn(preprocessing_pipe, len(time_cols), "time_processing.onnx")

    return dataset_bundle

def scale_route_lookup(cfg: Config, df: pd.DataFrame, train_hashes: set, aggregated: bool):
    route_features = list(cfg.dataset.route_feature_names)
    scaling_features = list(cfg.dataset.scaling_route_features)

    scaling_idx = [route_features.index(c) for c in scaling_features]
    passthrough_idx = [i for i, c in enumerate(route_features) if c not in scaling_features]

    train_df = df[df["route_seq_hash"].isin(train_hashes)]
    X_train = train_df[route_features].values.astype(np.float32)

    ct = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), scaling_idx),
            ("passthrough", "passthrough", passthrough_idx)
        ]
    )

    steps = []
    steps.append(("scaling", ct))
    if cfg.dataset.pca:
        steps.append(("pca", PCA(n_components=0.95)))
    pipe = Pipeline(steps)

    pipe.fit(X_train)

    X_all = df[route_features].values.astype(np.float32)
    transformed = pipe.transform(X_all)

    if cfg.dataset.pca:
        feature_names = [f"pca_route_{i}" for i in range(transformed.shape[1])]
    else:
        feature_names = scaling_features + [c for c in route_features if c not in scaling_features]

    ids = df.drop(columns=route_features)
    transformed_df = pd.DataFrame(transformed, index=df.index, columns=feature_names)
    processed_df = pd.concat([ids, transformed_df], axis=1)

    save_sklearn(pipe, len(route_features), f"{'aggr_' if aggregated else 'seq_'}route_processing.onnx")

    return processed_df

def pca_time_features(cfg: Config, dataset_bundle: DatasetBundle):
    time_cols = list(cfg.dataset.time_feature_names)
    pca = PCA(n_components=0.95)
    pca.fit(dataset_bundle.train.x[time_cols])

    for split_name in ["train", "val", "test"]:
        split = getattr(dataset_bundle, split_name)

        features = split.x[time_cols]
        ids = split.x.drop(columns=time_cols)

        X_pca = pca.transform(features)
        pca_cols = [f"pca_time_{i}" for i in range(X_pca.shape[1])]
        pca_df = pd.DataFrame(X_pca, index=features.index, columns=pca_cols)
        split.x = pd.concat([ids, pca_df], axis=1)

    return dataset_bundle, pca

def pca_route_lookup(cfg:Config, df: pd.DataFrame, train_hashes: set):
    route_feature_names = list(cfg.dataset.route_feature_names)
    train_df = df[df["route_seq_hash"].isin(train_hashes)]
    stacked_train_data = train_df[route_feature_names].values.astype(np.float32)
    pca = PCA(n_components=0.95)
    pca.fit(stacked_train_data)
    print(len(pca.components_), len(route_feature_names))

    features = df[route_feature_names]
    ids = df.drop(columns=route_feature_names)

    X_pca = pca.transform(features)
    pca_cols = [f"pca_route_{i}" for i in range(X_pca.shape[1])]
    pca_df = pd.DataFrame(X_pca, index=features.index, columns=pca_cols)
    df_reduced = pd.concat([ids, pca_df], axis=1)
    return df_reduced, pca

def train_sampler(df: pd.DataFrame, y: pd.Series):
    df["target"] = y.squeeze()
    max_target = df["target"].max()
    bins = list(range(0, 2001, 200))
    if max_target > 2000:
        bins = bins + [max_target + 1]

    df["target_bin"] = pd.cut(df["target"], bins=bins, labels=False, include_lowest=True)

    bin_counts = df["target_bin"].value_counts().sort_index()
    inv_freq = 1.0 / (bin_counts + 1e-6)

    inv_freq = inv_freq.clip(upper=10)
    sample_weights = df["target_bin"].map(inv_freq).values
    sample_weights_tensor = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=16_777_216,
        replacement=True
    )
    return sampler


def create_dataloader(cfg: Config, dataset_split: DatasetSplit, route_lookup, route_feature_indices, collate_fn, num_workers, train=False) -> DataLoader:
    if cfg.training.route_based_training and train:
        dataset = RouteBasedDataset(dataset_split, route_lookup, cfg.dataset.time_feature_names, route_feature_indices, cfg.training.random_state)
    else:
        dataset = MappingDataset(dataset_split, route_lookup, cfg.dataset.time_feature_names, route_feature_indices)
    sampler = None
    shuffle = True
    torch.manual_seed(cfg.training.random_state)
    data_loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, sampler=sampler)
    return data_loader


def create_dataloaders(cfg: Config, dataset_bundle: DatasetBundle, route_lookup, is_route_sequence, num_workers) -> Tuple[DataLoader, DataLoader, DataLoader]:
    route_feature_indices = [cfg.dataset.route_feature_names_full.index(name) for name in cfg.dataset.route_feature_names]

    if is_route_sequence:
        if cfg.training.route_based_training:
            train_collate_fn = route_based_seq_collate_fn
        else:
            train_collate_fn = seq_collate_fn
        val_collate_fn = seq_collate_fn
    else:
        if cfg.training.route_based_training:
            train_collate_fn = route_based_aggr_collate_fn
        else:
            train_collate_fn = aggr_collate_fn
        val_collate_fn = aggr_collate_fn

    train_loader = create_dataloader(cfg, dataset_bundle.train, route_lookup, route_feature_indices, train_collate_fn, num_workers, train=True)
    val_loader = create_dataloader(cfg, dataset_bundle.val, route_lookup, route_feature_indices, val_collate_fn, num_workers)
    test_loader = create_dataloader(cfg, dataset_bundle.test, route_lookup, route_feature_indices, val_collate_fn, num_workers)
    return train_loader, val_loader, test_loader
