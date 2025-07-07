import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

import config.paths as paths
from config.config import Config
from data.data_processing import scale_time_features, split_data, scale_seq_route_lookup, scale_aggr_route_lookup


def preprocess_splits(cfg, path):
    df = pd.read_csv(path + ".csv")
    dataset_bundle = split_data(cfg, df)
    dataset_bundle = scale_time_features(cfg, dataset_bundle)
    dataset_bundle.save(paths.DATASET_BUNDLE_DIR)
    train_hashes = set(dataset_bundle.train.x["route_seq_hash"].unique())
    return train_hashes

def create_route_dict(path, route_feature_names, train_hashes, scaling_fn):
    df = pd.read_csv(path + ".csv")
    route_data = {}
    for hash_val, group in tqdm(df.groupby("route_seq_hash")):
        route_data[str(hash_val)] = group[route_feature_names].values.astype(np.float32)

    print("Scaling...", flush=True)
    route_data = scaling_fn(route_data, train_hashes)

    with open(path + ".pkl", "wb") as f:
        pickle.dump(route_data, f)

def data_conversions(cfg: Config):
    print("Converting csv to parquet", flush=True)
    train_hashes = preprocess_splits(cfg, paths.DATASETS_DIR + cfg.dataset.time)
    print("Creating route sequence dict", flush=True)
    create_route_dict(paths.DATASETS_DIR + cfg.dataset.route_seq, cfg.dataset.route_feature_names, train_hashes, scaling_fn=scale_seq_route_lookup)
    print("Creating aggregated route dict", flush=True)
    create_route_dict(paths.DATASETS_DIR + cfg.dataset.route_aggr, cfg.dataset.route_feature_names, train_hashes, scaling_fn=scale_aggr_route_lookup)

