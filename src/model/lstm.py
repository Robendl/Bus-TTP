import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence

from config.config import Config


class LSTMFeedforwardCombination(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        lstm_input_dim = len(cfg.dataset.route_feature_names)
        lstm_hidden_dim = cfg.model.lstm.lstm_hidden_dim
        time_input_dim = len(cfg.dataset.time_feature_names)
        ff_hidden_dims = cfg.model.lstm.ff_hidden_dims
        dropout = cfg.model.lstm.dropout
        self.name = "LSTM"
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim, num_layers=cfg.model.lstm.num_lstm_layers, batch_first=True, dropout=dropout)
        self.lstm_dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(lstm_hidden_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden_list = nn.ModuleList()
        self.fc_hidden_list.append(nn.Linear(time_input_dim, ff_hidden_dims[0]))
        for i in range(len(ff_hidden_dims) - 1):
            self.fc_hidden_list.append(nn.Linear(ff_hidden_dims[i], ff_hidden_dims[i + 1]))

        self.final = nn.Sequential(
            nn.Linear(lstm_hidden_dim + ff_hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, inp):
        time_features, padded_routes, lengths = inp
        packed = pack_padded_sequence(padded_routes, lengths, batch_first=True, enforce_sorted=False)

        _, (hn, _) = self.lstm(packed)
        hn = hn.squeeze(0)
        hn = self.ln(hn)
        hn = self.lstm_dropout(hn)

        tf = time_features
        for hidden_layer in self.fc_hidden_list:
            tf = hidden_layer(tf)
            tf = self.relu(tf)
            tf = self.dropout(tf)

        combined = torch.cat((hn, tf), dim=1)
        output = self.final(combined)
        return output