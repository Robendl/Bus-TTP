from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

from data.dataset_bundle import DatasetBundle, DatasetSplit


class MappingDataset(Dataset):
    def __init__(
        self,
        dataset_split: DatasetSplit,
        route_lookup,
        time_feature_names,
        route_feature_names
    ):
        self.time_features = torch.tensor(
            dataset_split.x[time_feature_names].to_numpy(dtype=np.float32)
        )
        self.ids = dataset_split.x['id']
        self.labels = torch.tensor(
            dataset_split.y.to_numpy(dtype=np.float32)
        )
        self.route_seq_hashes = dataset_split.x["route_seq_hash"].values
        self.route_lookup = route_lookup

    def __len__(self):
        return len(self.time_features)

    def __getitem__(self, idx):
        time_feat = self.time_features[idx]
        label = self.labels[idx]
        id = self.ids[idx]

        route_seq_hash = self.route_seq_hashes[idx]
        route_sequence = self.route_lookup[route_seq_hash]
        route_tensor = torch.from_numpy(route_sequence)

        return id, (time_feat, route_tensor), label


def seq_collate_fn(batch):
    ids, features_tuple, labels_list = zip(*batch)
    time_features_list, route_sequences_list = zip(*features_tuple)

    time_features = torch.stack(time_features_list)
    labels = torch.stack(labels_list)

    lengths = torch.tensor([seq.size(0) for seq in route_sequences_list])
    padded_routes = pad_sequence(route_sequences_list, batch_first=True)

    return ids, (time_features, padded_routes, lengths), labels

def aggr_collate_fn(batch):
    ids, features_tuple, labels_list = zip(*batch)
    time_features_list, route_features_list = zip(*features_tuple)

    time_features = torch.stack(time_features_list)
    route_features = torch.stack(route_features_list).squeeze(1)
    full_features = torch.cat((time_features, route_features), dim=1)
    labels = torch.stack(labels_list)

    return ids, full_features, labels

# class SequenceDataset(Dataset):
#     def __init__(
#         self,
#         time_features_df,  # full DataFrame, will be split per worker
#         labels_series,
#         full_route_lookup,
#         time_feature_names,
#         route_feature_names,
#     ):
#         self.time_features_df = time_features_df
#         self.labels_series = labels_series
#         self.full_route_lookup = full_route_lookup
#         self.time_feature_names = time_feature_names
#         self.route_feature_names = route_feature_names
#
#         # These will be initialized per-worker
#         self.time_features = None
#         self.route_lookup = None
#         self.labels = None
#         self.route_seq_hashes = None
#
#     def _init_worker(self):
#         worker_info = get_worker_info()
#         if worker_info is None:
#             # single-process data loading
#             time_features = self.time_features_df
#             labels = self.labels_series
#
#         else:
#             # split data across workers
#             worker_id = worker_info.id
#             num_workers = worker_info.num_workers
#             total_len = len(self.time_features_df)
#             per_worker = total_len // num_workers
#             start = worker_id * per_worker
#             # ensure the last worker gets any remainder
#             end = total_len if worker_id == num_workers - 1 else (worker_id + 1) * per_worker
#
#             time_features = self.time_features_df.iloc[start:end].reset_index(drop=True)
#             labels = self.labels_series.iloc[start:end].reset_index(drop=True)
#
#         self.time_features = time_features[self.time_feature_names].to_numpy(dtype=np.float32)
#         self.labels = labels.to_numpy(dtype=np.float32)
#         self.route_seq_hashes = time_features["route_seq_hash"].values
#
#         unique_hashes = set(self.route_seq_hashes)
#         self.route_lookup = {
#             h: self.full_route_lookup[h]
#             for h in unique_hashes if h in self.full_route_lookup
#         }
#
#     def __len__(self):
#         if self.time_features is None:
#             self._init_worker()
#         return len(self.time_features)
#
#     def __getitem__(self, idx):
#         if self.time_features is None:
#             self._init_worker()
#
#         time_feat = torch.tensor(self.time_features[idx], dtype=torch.float32)
#         label = torch.tensor(self.labels[idx], dtype=torch.float32)
#
#         route_seq_hash = self.route_seq_hashes[idx]
#         route_features_seq = self.route_lookup[route_seq_hash]
#         route_tensor = torch.tensor(route_features_seq, dtype=torch.float32)
#
#         return (time_feat, route_tensor), label