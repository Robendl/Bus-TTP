import math
import pickle
from typing import Dict

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import torch
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from skl2onnx import convert_sklearn

import config.paths as paths
from config.config import Config
from data.data_processing import scale_time_features, scale_route_lookup, pca_route_lookup, \
    pca_time_features, create_dataset_bundle
from data.dataset_bundle import DatasetBundle
from plot.plot import plot_deviation


def csv_to_parquet(path, use_subset=False):
    if use_subset:
        df_train = pd.read_csv(path + "_train.csv", nrows=20000)
        df_test = pd.read_csv(path + "_test.csv", nrows=1000)
    else:
        df_train = pd.read_csv(path + "_train.csv")
        df_test = pd.read_csv(path + "_test.csv")
    df_train.to_parquet(path + "_train.parquet")
    df_test.to_parquet(path + "_test.parquet")

def iqr_filter(group, factor, column="recorded_elapsed_time"):
    q1 = group[column].quantile(0.25)
    q3 = group[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return group[(group[column] >= lower) & (group[column] <= upper)]

def preprocess_splits(cfg, path):
    # db = DatasetBundle.load(paths.DATASET_BUNDLE_DIR + ("_pca" if cfg.dataset.pca else ""))
    # return set(db.train.x["route_seq_hash"].unique())

    df_train = pd.read_parquet(path + "_train.parquet")
    df_test = pd.read_parquet(path + "_test.parquet")

    if cfg.dataset.include_mapping_errors:
        print("Including mapping errors")
        df_mapping_error = pd.read_csv(path + "_mapping_error.csv")
        df_train = pd.concat([df_train, df_mapping_error], ignore_index=True)

    if cfg.dataset.include_measurement_errors:
        print("Including measurement errors")
        df_measurement_error = pd.read_csv(path + "_measurement_error.csv")
        df_train = pd.concat([df_train, df_measurement_error], ignore_index=True)

    df_train["excess_circuity"] = np.log(1 + df_train["excess_circuity"])
    df_test["excess_circuity"] = np.log(1 + df_test["excess_circuity"])

    original_length = df_train.shape[0]
    df_train = df_train[df_train.groupby("route_seq_hash")["route_seq_hash"].transform("count") >= 4]
    if df_train.isnull().any().any():
        print("1: fWarning: NaNs in dataset")
    if cfg.dataset.filter_outliers:
        print("Filtering outliers")
        filtered_df_train = df_train.groupby("route_seq_hash", group_keys=False).apply(iqr_filter, factor=cfg.dataset.iqr_factor)
        filtered_df_test = df_test.groupby("route_seq_hash", group_keys=False).apply(iqr_filter, factor=cfg.dataset.iqr_factor)
    else:
        print("Not filtering outliers")
        filtered_df_train = df_train
        filtered_df_test = df_test

    if filtered_df_train.isnull().any().any():
        print("2: Warning: NaNs in dataset")
    # filtered_df["id"].to_csv(paths.DATASETS_DIR + "filtered_ids.csv", index=False)

    new_fraction = filtered_df_train.shape[0] / original_length
    plot_df = df_train.copy()
    plot_filtered_df = filtered_df_train.copy()
    plot_deviation(plot_df, plot_filtered_df, new_fraction, log_scale=True)
    plot_deviation(plot_df, plot_filtered_df, new_fraction, log_scale=False)

    # dataset_bundle = split_data(cfg, filtered_df_train)
    dataset_bundle = create_dataset_bundle(cfg, filtered_df_train, filtered_df_test)

    dataset_bundle = scale_time_features(cfg, dataset_bundle)

    dataset_bundle.save(paths.DATASET_BUNDLE_DIR
                        + ("_pca" if cfg.dataset.pca else "")
                        + ("_multi" if cfg.dataset.multi_run else ""))

    if cfg.dataset.process_metadata:
        full_df = pd.read_csv(paths.DATASETS_DIR + cfg.dataset.metadata + ".csv")
        # val_metadata = full_df[full_df["id"].isin(dataset_bundle.val.x["id"])]
        # val_metadata.to_parquet(paths.DATASETS_DIR + cfg.dataset.metadata + "_val.parquet")
        test_metadata = full_df[full_df["id"].isin(dataset_bundle.test.x["id"])]
        test_metadata.to_parquet(paths.DATASETS_DIR + cfg.dataset.metadata + "_test.parquet")

        geoms = pd.read_csv(paths.DATASETS_DIR + cfg.dataset.geoms + ".csv")
        # val_geoms = geoms[geoms["geom_id"].isin(val_metadata["geom_id"])]
        # val_geoms.to_parquet(paths.DATASETS_DIR + cfg.dataset.geoms + "_val.parquet")
        test_geoms = geoms[geoms["geom_id"].isin(test_metadata["geom_id"])]
        test_geoms.to_parquet(paths.DATASETS_DIR + cfg.dataset.geoms + "_test.parquet")

    train_hashes = set(dataset_bundle.train.x["route_seq_hash"].unique())
    return train_hashes

def create_route_dict(cfg: Config, path, train_hashes, aggregated=False):
    df = pd.read_csv(path + ".csv")
    if not aggregated:
        df.drop(columns=["seq"], inplace=True)
    df = scale_route_lookup(cfg, df, train_hashes, aggregated)

    df.to_parquet(path + ".parquet")
    if df.isnull().any().any():
        print(f"3: Warning: NaNs in dataset a={aggregated}")
    route_lookup = {}

    for hash_val, group in tqdm(df.groupby("route_seq_hash")):
        group.drop(columns=["route_seq_hash"], inplace=True)
        values = group.values.astype(np.float32)
        if aggregated:
            values = values.reshape(1, -1)
        route_lookup[str(hash_val)] = values

    with open(path + ("_pca" if cfg.dataset.pca else "") + ".pkl", "wb") as f:
        pickle.dump(route_lookup, f)

def data_conversions(cfg: Config):
    print("Converting csv to parquet", flush=True)
    csv_to_parquet(paths.DATASETS_DIR + cfg.dataset.time, use_subset=cfg.dataset.use_subset)
    train_hashes = preprocess_splits(cfg, paths.DATASETS_DIR + cfg.dataset.time)
    print("Creating route sequence dict", flush=True)
    # create_route_dict(cfg, paths.DATASETS_DIR + cfg.dataset.route_seq, train_hashes)
    print("Creating aggregated route dict", flush=True)
    # create_route_dict(cfg, paths.DATASETS_DIR + cfg.dataset.route_aggr, train_hashes, aggregated=True)

def load_route_lookup(path) -> Dict[str, np.ndarray]:
    with open(path + ".pkl", "rb") as f:
        route_lookup = pickle.load(f)
    return route_lookup