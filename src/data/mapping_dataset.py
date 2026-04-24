"""Map-style PyTorch dataset and collate functions for OD-pair training.

Each item joins a trip's tabular features with its route's pre-computed
feature lookup. Two collate functions are provided:

- ``aggr_collate_fn``: route is represented as a single aggregated feature
  vector (used by the MLP).
- ``seq_collate_fn``: route is represented as a variable-length sequence
  of segment-level features (used by the LSTM).
"""
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from data.dataset_bundle import DatasetSplit


class MappingDataset(Dataset):
    def __init__(
        self,
        dataset_split: DatasetSplit,
        route_lookup,
        time_feature_names,
        route_feature_indices,
    ):
        feature_df = dataset_split.x.drop(["id", "route_seq_hash", "stop_to_stop_id"], axis=1)
        self.time_features = torch.tensor(feature_df.to_numpy(dtype=np.float32))
        self.labels = torch.tensor(dataset_split.y.to_numpy(dtype=np.float32))

        self.ids = dataset_split.x["id"]
        self.route_seq_hashes = dataset_split.x["route_seq_hash"].values
        self.stop_to_stop_ids = dataset_split.x["stop_to_stop_id"].values
        self.route_lookup = route_lookup
        self.route_feature_indices = route_feature_indices

    def __len__(self):
        return len(self.time_features)

    def __getitem__(self, idx):
        route_sequence = self.route_lookup[self.route_seq_hashes[idx]]
        route_tensor = torch.from_numpy(route_sequence)
        return self.ids[idx], (self.time_features[idx], route_tensor), self.labels[idx]


def seq_collate_fn(batch):
    ids, features, labels = zip(*batch)
    time_features, route_sequences = zip(*features)

    time_features = torch.stack(time_features)
    labels = torch.stack(labels)
    lengths = torch.tensor([seq.size(0) for seq in route_sequences])
    padded_routes = pad_sequence(route_sequences, batch_first=True)

    return ids, (time_features, padded_routes, lengths), labels


def aggr_collate_fn(batch):
    ids, features, labels = zip(*batch)
    time_features, route_features = zip(*features)

    time_features = torch.stack(time_features)
    route_features = torch.stack(route_features).squeeze(1)
    labels = torch.stack(labels)
    full_features = torch.cat((time_features, route_features), dim=1)

    return ids, full_features, labels
