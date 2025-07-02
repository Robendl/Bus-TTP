import pickle
import pandas as pd
import numpy as np
import torch
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

    with open(path + ".pkl", "wb") as f:
        pickle.dump(route_data, f)

def create_route_tensor(path, route_feature_names):
    route_seq = pd.read_parquet(path + ".parquet")
    # route_seq_sorted = route_seq.sort_values(by=["route_seq_id"]).copy()
    # grouped = route_seq_sorted.groupby("route_seq_id")
    #
    # route_tensors = [
    #     torch.tensor(group[route_feature_names].values, dtype=torch.float32)
    #     for _, group in tqdm(grouped)
    # ]
    grouped = route_seq.groupby("route_seq_id")

    max_len = grouped.size().max()
    D_route = route_seq[route_feature_names].shape[1]

    # Allocate padded tensor
    route_tensor_padded = torch.zeros((len(grouped), max_len, D_route), dtype=torch.float32)

    for route_id, group in tqdm(grouped):
        seq = torch.tensor(group[route_feature_names].values, dtype=torch.float32)
        route_tensor_padded[route_id, :seq.size(0)] = seq

    # Move to GPU

    torch.save(route_tensor_padded, path + "_test.pt")



def data_conversions(cfg: Config):
    # print("Converting csv to parquet", flush=True)
    # csv_to_parquet(paths.DATASETS_DIR + cfg.dataset.time)
    # csv_to_parquet(paths.DATASETS_DIR + cfg.dataset.route_seq)
    # csv_to_parquet(paths.DATASETS_DIR + cfg.dataset.route_aggr)
    print("Creating route tensor", flush=True)
    create_route_tensor(paths.DATASETS_DIR + cfg.dataset.route_seq, cfg.training.route_feature_names)

    # print("Creating route sequence dict", flush=True)
    # create_route_dict(paths.DATASETS_DIR + cfg.dataset.route_seq, cfg.training.route_feature_names)
    # print("Creating aggregated route dict", flush=True)
    # create_route_dict(paths.DATASETS_DIR + cfg.dataset.route_aggr, cfg.training.route_feature_names)

if __name__ == "__main__":
    csv_to_parquet("data/seq_dataset_rf")
    csv_to_parquet("data/seq_dataset_tf")
    # create_route_dict("data/seq_dataset_rf")
