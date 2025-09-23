import os

import torch
import pandas as pd
import numpy as np
from typing import Tuple, Dict

from hydra.core.hydra_config import HydraConfig
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

def create_dataset_bundle(cfg: Config, df_train: pd.DataFrame, df_test: pd.DataFrame) -> DatasetBundle:
    y_train = df_train["recorded_elapsed_time"]
    X_train = df_train.drop(columns=["recorded_elapsed_time"])
    y_test = df_test["recorded_elapsed_time"]
    X_test = df_test.drop(columns=["recorded_elapsed_time"])

    if cfg.dataset.use_validation:
        unique_ids = X_train['stop_to_stop_id'].unique()
        test_ids = X_test['stop_to_stop_id'].unique()
        rng = np.random.default_rng(seed=cfg.training.random_state)
        shuffled_ids = rng.permutation(unique_ids)
        n_total = len(shuffled_ids) + len(test_ids)
        n_val = int(n_total * cfg.training.val_size)
        val_ids = shuffled_ids[:n_val]
        train_ids = shuffled_ids[n_val:]

        train_mask = X_train['stop_to_stop_id'].isin(train_ids)
        val_mask = X_train['stop_to_stop_id'].isin(val_ids)
        X_train_subset, X_val = X_train[train_mask], X_train[val_mask]
        y_train_subset, y_val = y_train[train_mask], y_train[val_mask]
        train = DatasetSplit(X_train_subset, y_train_subset)
        val = DatasetSplit(X_val, y_val)
    else:
        train = DatasetSplit(X_train, y_train)
        val = None

    # if not cfg.dataset.multi_run:
    #     X_train.drop(columns=["stop_to_stop_id"], inplace=True)
    #     X_test.drop(columns=["stop_to_stop_id"], inplace=True)

    return DatasetBundle(
        train=train,
        val=val,
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
        pipeline_steps.append(("pca", PCA(n_components=cfg.dataset.n_components)))

    preprocessing_pipe = Pipeline(pipeline_steps)

    preprocessing_pipe.fit(dataset_bundle.train.x[time_cols].values)

    if cfg.dataset.use_validation:
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

    if cfg.dataset.pca:
        output_dir = HydraConfig.get().run.dir
        pca = preprocessing_pipe.named_steps["pca"]
        n_features_after = pca.n_components_
        explained = pca.explained_variance_ratio_
        cumulative = explained.cumsum()
        with open(f"{output_dir}/time_pca.txt", "w") as f:
            f.write(f"route before: {len(time_cols)}, after: {n_features_after} \n")
            f.write(f"Cumulative: {cumulative} \n")
            f.write(f"Total variance: {cumulative[-1]} \n")
            f.write(f"Explained: {explained} \n")

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
        steps.append(("pca", PCA(n_components=cfg.dataset.n_components)))
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

    if cfg.dataset.pca:
        output_dir = HydraConfig.get().run.dir
        pca = pipe.named_steps["pca"]
        n_features_after = pca.n_components_
        explained = pca.explained_variance_ratio_
        cumulative = explained.cumsum()
        with open(f"{output_dir}/route_{"aggr" if aggregated else "seq"}_pca.txt", "w") as f:
            f.write(f"route before: {len(route_features)}, after: {n_features_after} \n")
            f.write(f"Cumulative: {cumulative} \n")
            f.write(f"Total variance: {cumulative[-1]} \n")
            f.write(f"Explained: {explained} \n")

    return processed_df

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
    if cfg.dataset.use_validation:
        val_loader = create_dataloader(cfg, dataset_bundle.val, route_lookup, route_feature_indices, val_collate_fn, num_workers)
    else:
        val_loader = None
    test_loader = create_dataloader(cfg, dataset_bundle.test, route_lookup, route_feature_indices, val_collate_fn, num_workers)
    return train_loader, val_loader, test_loader
