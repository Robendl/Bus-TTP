from itertools import chain

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np
import random

class RouteBasedDataset(Dataset):
    def __init__(
        self,
        dataset_split,
        route_lookup,
        time_feature_names,
        route_feature_indices,
        n_time_samples_per_route=4,
    ):
        self.n_time_samples = n_time_samples_per_route

        self.time_features = torch.tensor(
            dataset_split.x[time_feature_names].to_numpy(dtype=np.float32)
        )
        self.labels = torch.tensor(
            dataset_split.y.to_numpy(dtype=np.float32)
        )
        self.ids = dataset_split.x["id"].values
        self.route_seq_hashes_full = dataset_split.x["route_seq_hash"].values

        self.route_lookup = route_lookup
        self.route_feature_indices = route_feature_indices

        # Index: route_hash → list of time-row indices
        self.route_to_indices = {}
        for idx, route_hash in enumerate(self.route_seq_hashes_full):
            if route_hash not in self.route_to_indices:
                self.route_to_indices[route_hash] = []
            self.route_to_indices[route_hash].append(idx)

        self.unique_routes = list(self.route_to_indices.keys())

    def __len__(self):
        return len(self.unique_routes)

    def __getitem__(self, idx):
        route_hash = self.unique_routes[idx]
        time_indices = self.route_to_indices[route_hash]

        sampled_indices = random.choices(time_indices, k=self.n_time_samples)

        time_feat_batch = self.time_features[sampled_indices]        # shape [3, D]
        label_batch = self.labels[sampled_indices]                   # shape [3]
        id_samples = self.ids[sampled_indices]                       # shape [3]

        # Route features
        route_sequence = self.route_lookup[route_hash][:, self.route_feature_indices]
        route_tensor = torch.from_numpy(route_sequence)

        return id_samples, (time_feat_batch, route_tensor), label_batch

def route_based_seq_collate_fn(batch):
    ids_list, features_tuple, labels_list = zip(*batch)
    time_features_batch, route_sequences = zip(*features_tuple)

    # time_features_batch: [B] list of [3, D]
    # labels_list:         [B] list of [3]

    time_features = torch.cat(time_features_batch, dim=0)  # [B*3, D]
    labels = torch.cat(labels_list, dim=0)  # [B*3]
    ids = list(chain.from_iterable(ids_list))

    repeated_routes = []
    lengths = []
    for route in route_sequences:
        repeated_routes.extend([route] * 4)
        lengths.extend([route.size(0)] * 4)

    padded_routes = pad_sequence(repeated_routes, batch_first=True)  # [B*3, max_L, D]
    lengths = torch.tensor(lengths)


    # print(ids.shape, time_features.shape, padded_routes.shape, lengths.shape, labels.shape)
    return ids, (time_features, padded_routes, lengths), labels


def route_based_aggr_collate_fn(batch):
    ids_list, features_tuple, labels_list = zip(*batch)
    time_features_batch, route_features_list = zip(*features_tuple)

    time_features = torch.cat(time_features_batch, dim=0)  # [B*3, D_time]
    labels = torch.cat(labels_list, dim=0)  # [B*3]
    ids = list(chain.from_iterable(ids_list))

    expanded_route_features = torch.cat([
        route_feat.expand(4, -1) for route_feat in route_features_list
    ], dim=0)

    full_features = torch.cat((time_features, expanded_route_features), dim=1)

    # print(len(ids), full_features.shape, labels.shape)
    return ids, full_features, labels
