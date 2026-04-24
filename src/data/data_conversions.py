"""One-shot data preparation: CSV → parquet, IQR filtering, OD-disjoint split,
scaling, and per-route lookup pickling.

Run once before training via ``cfg.pre_data_conversions=True``.
"""
import pickle
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import config.paths as paths
from config.config import Config
from data.data_processing import create_dataset_bundle, scale_route_lookup, scale_time_features
from plot.plot import plot_deviation


def csv_to_parquet(path: str, use_subset: bool = False) -> None:
    nrows_train = 20_000 if use_subset else None
    nrows_test = 1_000 if use_subset else None

    pd.read_csv(path + "_train.csv", nrows=nrows_train).to_parquet(path + "_train.parquet")
    pd.read_csv(path + "_test.csv", nrows=nrows_test).to_parquet(path + "_test.parquet")


def iqr_filter(group: pd.DataFrame, factor: float, column: str = "recorded_elapsed_time") -> pd.DataFrame:
    q1, q3 = group[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return group[(group[column] >= lower) & (group[column] <= upper)]


def split_dataframe(cfg: Config, df: pd.DataFrame):
    """OD-disjoint split: every stop-to-stop pair is fully in train or fully in test."""
    unique_ids = df["stop_to_stop_id"].unique()
    rng = np.random.default_rng(seed=cfg.training.random_state)
    shuffled_ids = rng.permutation(unique_ids)
    n_test = int(len(unique_ids) * cfg.training.test_size)
    test_ids = set(shuffled_ids[:n_test])
    train_ids = set(shuffled_ids[n_test:])

    return df[df["stop_to_stop_id"].isin(train_ids)], df[df["stop_to_stop_id"].isin(test_ids)]


def _maybe_inject_error_subset(cfg: Config, base_path: str, name: str, df_train, df_test):
    extra = pd.read_csv(f"{base_path}_{name}.csv")
    extra_train, extra_test = split_dataframe(cfg, extra)
    df_train = pd.concat([df_train, extra_train], ignore_index=True)
    df_test = pd.concat([df_test, extra_test], ignore_index=True)
    return df_train, df_test


def preprocess_splits(cfg: Config, path: str) -> set:
    df_train = pd.read_parquet(path + "_train.parquet")
    df_test = pd.read_parquet(path + "_test.parquet")

    if not cfg.dataset.use_test:
        df_train = pd.concat([df_train, df_test], ignore_index=True)

    if cfg.dataset.include_mapping_errors:
        print("Including mapping errors")
        df_train, df_test = _maybe_inject_error_subset(cfg, path, "mapping_error", df_train, df_test)
    if cfg.dataset.include_measurement_errors:
        print("Including measurement errors")
        df_train, df_test = _maybe_inject_error_subset(cfg, path, "measurement_error", df_train, df_test)
    if cfg.dataset.include_invalid:
        print("Including invalid trips")
        df_train, df_test = _maybe_inject_error_subset(cfg, path, "invalid", df_train, df_test)

    df_train["excess_circuity"] = np.log1p(df_train["excess_circuity"])
    df_test["excess_circuity"] = np.log1p(df_test["excess_circuity"])

    original_length = df_train.shape[0]
    df_train = df_train[df_train.groupby("route_seq_hash")["route_seq_hash"].transform("count") >= 4]

    if cfg.dataset.filter_outliers:
        print("Filtering outliers")
        filtered_train = df_train.groupby("route_seq_hash", group_keys=False).apply(
            iqr_filter, factor=cfg.dataset.iqr_factor
        )
        filtered_test = df_test.groupby("route_seq_hash", group_keys=False).apply(
            iqr_filter, factor=cfg.dataset.iqr_factor
        )
    else:
        filtered_train, filtered_test = df_train, df_test

    new_fraction = filtered_train.shape[0] / original_length
    plot_deviation(df_train.copy(), filtered_train.copy(), new_fraction, log_scale=True)
    plot_deviation(df_train.copy(), filtered_train.copy(), new_fraction, log_scale=False)

    dataset_bundle = create_dataset_bundle(cfg, filtered_train, filtered_test)
    if cfg.dataset.scale_features:
        dataset_bundle = scale_time_features(cfg, dataset_bundle)

    dataset_bundle.save(paths.DATASET_BUNDLE_DIR, cfg)

    if cfg.dataset.process_metadata:
        full_df = pd.read_csv(paths.DATASETS_DIR + cfg.dataset.metadata + ".csv")
        test_metadata = full_df[full_df["id"].isin(dataset_bundle.test.x["id"])]
        test_metadata.to_parquet(paths.DATASETS_DIR + cfg.dataset.metadata + "_test.parquet")

        geoms = pd.read_csv(paths.DATASETS_DIR + cfg.dataset.geoms + ".csv")
        test_geoms = geoms[geoms["geom_id"].isin(test_metadata["geom_id"])]
        test_geoms.to_parquet(paths.DATASETS_DIR + cfg.dataset.geoms + "_test.parquet")

    return set(dataset_bundle.train.x["route_seq_hash"].unique())


def _route_lookup_path(cfg: Config, base_path: str) -> str:
    return (
        base_path
        + ("_val" if cfg.dataset.use_validation else "")
        + ("_pca" if cfg.dataset.pca else "")
        + ("_fulltrain" if not cfg.dataset.use_test else "")
        + ("_multi" if cfg.dataset.multi_run else "")
    )


def create_route_dict(cfg: Config, base_path: str, train_hashes: set, aggregated: bool = False) -> None:
    df = pd.read_csv(base_path + ".csv")
    if not aggregated:
        df.drop(columns=["seq"], inplace=True)

    if cfg.dataset.scale_features:
        df = scale_route_lookup(cfg, df, train_hashes, aggregated)

    out_path = _route_lookup_path(cfg, base_path)
    df.to_parquet(out_path + ".parquet")

    route_lookup: Dict[str, np.ndarray] = {}
    for hash_val, group in tqdm(df.groupby("route_seq_hash"), desc="Building route lookup"):
        values = group.drop(columns=["route_seq_hash"]).values.astype(np.float32)
        if aggregated:
            values = values.reshape(1, -1)
        route_lookup[str(hash_val)] = values

    with open(out_path + ".pkl", "wb") as f:
        pickle.dump(route_lookup, f)


def data_conversions(cfg: Config) -> None:
    print("Splitting and filtering trip data")
    train_hashes = preprocess_splits(cfg, paths.DATASETS_DIR + cfg.dataset.time)
    print("Creating route sequence dict")
    create_route_dict(cfg, paths.DATASETS_DIR + cfg.dataset.route_seq, train_hashes)
    print("Creating aggregated route dict")
    create_route_dict(cfg, paths.DATASETS_DIR + cfg.dataset.route_aggr, train_hashes, aggregated=True)


def load_route_lookup(cfg: Config, base_path: str) -> Dict[str, np.ndarray]:
    path = _route_lookup_path(cfg, base_path)
    with open(path + ".pkl", "rb") as f:
        return pickle.load(f)
