from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
class SequenceDataset(Dataset):
    def __init__(
        self,
        df_time: pd.DataFrame,
        df_labels: pd.DataFrame,
        route_lookup,
        time_feature_names,
        route_feature_names,
        device
    ):
        self.time_features = df_time[time_feature_names].to_numpy(dtype=np.float32)
        self.labels = df_labels.to_numpy(dtype=np.float32)
        self.route_seq_hashes = df_time["route_seq_hash"].values
        self.route_lookup = route_lookup
        self.device = device

    def __len__(self):
        return len(self.time_features)

    def __getitem__(self, idx):
        time_feat = torch.tensor(self.time_features[idx], dtype=torch.float32).to(self.device)
        label = torch.tensor(self.labels[idx], dtype=torch.float32).to(self.device)

        route_seq_hash = self.route_seq_hashes[idx]
        route_features_seq = self.route_lookup[route_seq_hash]
        route_tensor = torch.tensor(route_features_seq, dtype=torch.float32).to(self.device)

        return (time_feat, route_tensor), label


def collate_fn(batch):
    features_tuple, labels_list = zip(*batch)
    time_features_list, route_sequences_list = zip(*features_tuple)
    time_features = torch.stack(time_features_list)    # [B, T]
    labels = torch.stack(labels_list)                  # [B, 1] or [B]

    lengths = torch.tensor([seq.size(0) for seq in route_sequences_list])  # [B]
    padded_routes = pad_sequence(route_sequences_list, batch_first=True)   # [B, max_seq_len_in_batch, F]

    return (time_features, padded_routes, lengths), labels