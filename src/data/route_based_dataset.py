"""Route-based sampling dataset.

Instead of yielding individual trips, this dataset yields one *route* at a
time and samples a small number of trips from that route. This biases each
training batch to contain richer per-route variation, which empirically
helped the LSTM learn route representations.
"""
import random
from itertools import chain

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class RouteBasedDataset(Dataset):
    def __init__(
        self,
        dataset_split,
        route_lookup,
        time_feature_names,
        route_feature_indices,
        random_state,
        n_time_samples_per_route: int = 4,
    ):
        self.epoch_seed = 0
        self.random_state = random_state
        self.n_time_samples = n_time_samples_per_route

        feature_df = dataset_split.x.drop(["id", "route_seq_hash", "stop_to_stop_id"], axis=1)
        self.time_features = torch.tensor(feature_df.to_numpy(dtype=np.float32))
        self.labels = torch.tensor(dataset_split.y.to_numpy(dtype=np.float32))
        self.ids = dataset_split.x["id"].values
        self.route_seq_hashes_full = dataset_split.x["route_seq_hash"].values
        self.stop_to_stop_ids = dataset_split.x["stop_to_stop_id"].values

        self.route_lookup = route_lookup
        self.route_feature_indices = route_feature_indices

        self.route_to_indices: dict[str, list[int]] = {}
        for idx, route_hash in enumerate(self.route_seq_hashes_full):
            self.route_to_indices.setdefault(route_hash, []).append(idx)
        self.unique_routes = list(self.route_to_indices.keys())

    def __len__(self):
        return len(self.unique_routes)

    def set_epoch_seed(self, epoch: int):
        self.epoch_seed = (epoch + 1) * self.random_state % 1e5

    def __getitem__(self, idx):
        route_hash = self.unique_routes[idx]
        time_indices = self.route_to_indices[route_hash]

        random.seed(self.epoch_seed)
        sampled = random.choices(time_indices, k=self.n_time_samples)

        time_feat_batch = self.time_features[sampled]
        label_batch = self.labels[sampled]
        id_samples = self.ids[sampled]
        route_tensor = torch.from_numpy(self.route_lookup[route_hash])

        return id_samples, (time_feat_batch, route_tensor), label_batch


def route_based_seq_collate_fn(batch):
    ids_list, features, labels_list = zip(*batch)
    time_features_batch, route_sequences = zip(*features)

    time_features = torch.cat(time_features_batch, dim=0)
    labels = torch.cat(labels_list, dim=0)
    ids = list(chain.from_iterable(ids_list))

    samples_per_route = time_features_batch[0].shape[0]
    repeated_routes = []
    lengths = []
    for route in route_sequences:
        repeated_routes.extend([route] * samples_per_route)
        lengths.extend([route.size(0)] * samples_per_route)
    padded_routes = pad_sequence(repeated_routes, batch_first=True)
    lengths = torch.tensor(lengths)

    return ids, (time_features, padded_routes, lengths), labels


def route_based_aggr_collate_fn(batch):
    ids_list, features, labels_list = zip(*batch)
    time_features_batch, route_features_list = zip(*features)

    samples_per_route = time_features_batch[0].shape[0]
    time_features = torch.cat(time_features_batch, dim=0)
    labels = torch.cat(labels_list, dim=0)
    ids = list(chain.from_iterable(ids_list))

    expanded_route_features = torch.cat(
        [route.expand(samples_per_route, -1) for route in route_features_list],
        dim=0,
    )
    full_features = torch.cat((time_features, expanded_route_features), dim=1)

    return ids, full_features, labels
