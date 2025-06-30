from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

from data.dataset_bundle import DatasetBundle, DatasetSplit


class SequenceDataset(Dataset):
    def __init__(
        self,
        dataset_split: DatasetSplit,
        route_lookup,
        time_feature_names,
        route_feature_names,
        device
    ):
        self.time_features = dataset_split.x[time_feature_names].to_numpy(dtype=np.float32)
        self.labels = dataset_split.y.to_numpy(dtype=np.float32)
        self.route_seq_hashes = dataset_split.x["route_seq_hash"].values
        self.route_lookup = route_lookup
        self.device = device

    def __len__(self):
        return len(self.time_features)

    def __getitem__(self, idx):
        time_feat = torch.tensor(self.time_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        route_seq_hash = self.route_seq_hashes[idx]
        route_features_seq = self.route_lookup[route_seq_hash]
        route_tensor = torch.tensor(route_features_seq, dtype=torch.float32)

        return (time_feat, route_tensor), label


class CollateFn:
    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        features_tuple, labels_list = zip(*batch)
        time_features_list, route_sequences_list = zip(*features_tuple)

        time_features = torch.stack(time_features_list)
        labels = torch.stack(labels_list)

        lengths = torch.tensor([seq.size(0) for seq in route_sequences_list])
        padded_routes = pad_sequence(route_sequences_list, batch_first=True)

        return (time_features.to(self.device), padded_routes.to(self.device), lengths), labels.to(self.device)
