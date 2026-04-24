"""Dataset assembly, scaling pipelines, and PyTorch DataLoader factories."""
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import paths
from config.config import Config
from data.dataset_bundle import DatasetBundle, DatasetSplit
from data.mapping_dataset import MappingDataset, aggr_collate_fn, seq_collate_fn
from data.route_based_dataset import (
    RouteBasedDataset,
    route_based_aggr_collate_fn,
    route_based_seq_collate_fn,
)


def create_dataset_bundle(cfg: Config, df_train: pd.DataFrame, df_test: pd.DataFrame) -> DatasetBundle:
    """Build a DatasetBundle, optionally carving an OD-disjoint validation slice off train."""
    y_train = df_train["recorded_elapsed_time"]
    X_train = df_train.drop(columns=["recorded_elapsed_time"])
    y_test = df_test["recorded_elapsed_time"]
    X_test = df_test.drop(columns=["recorded_elapsed_time"])

    val = None
    if cfg.dataset.use_validation:
        unique_train_ids = X_train["stop_to_stop_id"].unique()
        n_total = len(unique_train_ids) + X_test["stop_to_stop_id"].nunique()
        n_val = int(n_total * cfg.training.val_size)

        rng = np.random.default_rng(seed=cfg.training.random_state)
        shuffled = rng.permutation(unique_train_ids)
        val_ids = set(shuffled[:n_val])
        train_ids = set(shuffled[n_val:])

        train_mask = X_train["stop_to_stop_id"].isin(train_ids)
        val_mask = X_train["stop_to_stop_id"].isin(val_ids)
        train = DatasetSplit(X_train[train_mask], y_train[train_mask])
        val = DatasetSplit(X_train[val_mask], y_train[val_mask])
    else:
        train = DatasetSplit(X_train, y_train)

    test = DatasetSplit(X_test, y_test) if cfg.dataset.use_test else None
    return DatasetBundle(train=train, val=val, test=test)


def _build_scaling_pipeline(
    feature_names: list[str], scaling_features: list[str], cfg: Config
) -> Pipeline:
    scaling_idx = [feature_names.index(c) for c in scaling_features]
    passthrough_idx = [i for i, c in enumerate(feature_names) if c not in scaling_features]

    column_transformer = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), scaling_idx),
            ("passthrough", "passthrough", passthrough_idx),
        ]
    )
    steps = [("scaling", column_transformer)]
    if cfg.dataset.pca:
        steps.append(("pca", PCA(n_components=cfg.dataset.n_components)))
    return Pipeline(steps)


def _save_pca_summary(pipeline: Pipeline, n_input_features: int, filename: str) -> None:
    output_dir = HydraConfig.get().run.dir
    pca = pipeline.named_steps["pca"]
    explained = pca.explained_variance_ratio_
    cumulative = explained.cumsum()
    with open(f"{output_dir}/{filename}", "w") as f:
        f.write(f"input features: {n_input_features}, components: {pca.n_components_}\n")
        f.write(f"cumulative explained variance: {cumulative}\n")
        f.write(f"total variance retained: {cumulative[-1]}\n")
        f.write(f"explained variance per component: {explained}\n")


def _save_sklearn_to_onnx(sklearn_obj, n_features: int, filename: str) -> None:
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(sklearn_obj, initial_types=initial_type)
    onnx_dir = f"{paths.RESULTS_DIR}/onnx"
    os.makedirs(onnx_dir, exist_ok=True)
    with open(f"{onnx_dir}/{filename}", "wb") as f:
        f.write(onnx_model.SerializeToString())


def scale_time_features(cfg: Config, dataset_bundle: DatasetBundle) -> DatasetBundle:
    time_cols = list(cfg.dataset.time_feature_names)
    scaling_cols = list(cfg.dataset.scaling_time_features)

    pipeline = _build_scaling_pipeline(time_cols, scaling_cols, cfg)
    pipeline.fit(dataset_bundle.train.x[time_cols].values)

    splits = [dataset_bundle.train]
    if cfg.dataset.use_validation:
        splits.append(dataset_bundle.val)
    if cfg.dataset.use_test:
        splits.append(dataset_bundle.test)

    for split in splits:
        features = split.x[time_cols].values
        ids = split.x.drop(columns=time_cols)
        processed = pipeline.transform(features)

        if cfg.dataset.pca:
            new_cols = [f"pca_time_{i}" for i in range(processed.shape[1])]
        else:
            new_cols = scaling_cols + [c for c in time_cols if c not in scaling_cols]

        processed_df = pd.DataFrame(processed, index=ids.index, columns=new_cols)
        split.x = pd.concat([ids, processed_df], axis=1)

    _save_sklearn_to_onnx(pipeline, len(time_cols), "time_processing.onnx")
    if cfg.dataset.pca:
        _save_pca_summary(pipeline, len(time_cols), "time_pca.txt")

    return dataset_bundle


def scale_route_lookup(cfg: Config, df: pd.DataFrame, train_hashes: set, aggregated: bool) -> pd.DataFrame:
    route_features = list(cfg.dataset.route_feature_names)
    scaling_features = list(cfg.dataset.scaling_route_features)

    pipeline = _build_scaling_pipeline(route_features, scaling_features, cfg)
    train_subset = df[df["route_seq_hash"].isin(train_hashes)]
    pipeline.fit(train_subset[route_features].values.astype(np.float32))

    transformed = pipeline.transform(df[route_features].values.astype(np.float32))

    if cfg.dataset.pca:
        feature_names = [f"pca_route_{i}" for i in range(transformed.shape[1])]
    else:
        feature_names = scaling_features + [c for c in route_features if c not in scaling_features]

    ids = df.drop(columns=route_features)
    transformed_df = pd.DataFrame(transformed, index=df.index, columns=feature_names)

    suffix = "aggr" if aggregated else "seq"
    _save_sklearn_to_onnx(pipeline, len(route_features), f"{suffix}_route_processing.onnx")
    if cfg.dataset.pca:
        _save_pca_summary(pipeline, len(route_features), f"route_{suffix}_pca.txt")

    return pd.concat([ids, transformed_df], axis=1)


def _create_dataloader(
    cfg: Config,
    dataset_split: DatasetSplit,
    route_lookup,
    route_feature_indices,
    collate_fn,
    num_workers: int,
    train: bool = False,
) -> DataLoader:
    if cfg.training.route_based_training and train:
        dataset = RouteBasedDataset(
            dataset_split, route_lookup, cfg.dataset.time_feature_names,
            route_feature_indices, cfg.training.random_state,
        )
    else:
        dataset = MappingDataset(
            dataset_split, route_lookup, cfg.dataset.time_feature_names, route_feature_indices,
        )

    torch.manual_seed(cfg.training.random_state)
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_dataloaders(
    cfg: Config,
    dataset_bundle: DatasetBundle,
    route_lookup,
    is_route_sequence: bool,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader | None, DataLoader | None]:
    route_feature_indices = [
        cfg.dataset.route_feature_names_full.index(name) for name in cfg.dataset.route_feature_names
    ]

    if is_route_sequence:
        train_collate = route_based_seq_collate_fn if cfg.training.route_based_training else seq_collate_fn
        eval_collate = seq_collate_fn
    else:
        train_collate = route_based_aggr_collate_fn if cfg.training.route_based_training else aggr_collate_fn
        eval_collate = aggr_collate_fn

    train_loader = _create_dataloader(
        cfg, dataset_bundle.train, route_lookup, route_feature_indices,
        train_collate, num_workers, train=True,
    )
    val_loader = (
        _create_dataloader(cfg, dataset_bundle.val, route_lookup, route_feature_indices, eval_collate, num_workers)
        if cfg.dataset.use_validation else None
    )
    test_loader = (
        _create_dataloader(cfg, dataset_bundle.test, route_lookup, route_feature_indices, eval_collate, num_workers)
        if cfg.dataset.use_test else None
    )
    return train_loader, val_loader, test_loader
