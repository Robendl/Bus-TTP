import pandas as pd
import numpy as np
from tqdm import tqdm

import config.paths as paths
from config.config import Config


def csv_to_parquet(path):
    df = pd.read_csv(path + ".csv")
    df.to_parquet(path + ".parquet", compression="snappy")

def create_route_dict(path, route_feature_names):
    df = pd.read_csv(path + ".csv")
    route_data = {}
    for hash_val, group in tqdm(df.groupby("route_seq_hash")):
        route_data[str(hash_val)] = group[route_feature_names].values.astype(np.float32)

    np.savez_compressed(path + ".npz", **route_data)

def data_conversions(cfg: Config):
    print("Converting csv to parquet")
    csv_to_parquet(paths.DATASETS_DIR + cfg.dataset.time)
    print("Creating route dict")
    create_route_dict(paths.DATASETS_DIR + cfg.dataset.route_seq, cfg.training.route_feature_names)


if __name__ == "__main__":
    csv_to_parquet("data/seq_dataset_rf")
    csv_to_parquet("data/seq_dataset_tf")
    # create_route_dict("data/seq_dataset_rf")
