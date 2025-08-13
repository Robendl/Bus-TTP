import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence

from config.config import Config


class LSTMFeedforwardCombination(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        lstm_input_dim = len(cfg.dataset.route_feature_names)
        lstm_hidden_dim = cfg.model.lstm.hidden_dim
        time_input_dim = len(cfg.dataset.time_feature_names)
        ff_hidden_dim = cfg.model.lstm.ff_hidden_dim
        dropout = cfg.model.lstm.dropout
        self.name = "LSTM"
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim, num_layers=1, batch_first=True)
        self.lstm_dropout = nn.Dropout(dropout)
        self.time_fc = nn.Sequential(
            nn.Linear(time_input_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.final = nn.Sequential(
            nn.Linear(lstm_hidden_dim + ff_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, inp):
        time_features, padded_routes, lengths = inp
        packed = pack_padded_sequence(padded_routes, lengths, batch_first=True, enforce_sorted=False)

        _, (hn, _) = self.lstm(packed)
        hn = hn.squeeze(0)
        hn = self.lstm_dropout(hn)

        time_emb = self.time_fc(time_features)

        combined = torch.cat((hn, time_emb), dim=1)
        output = self.final(combined)
        return output