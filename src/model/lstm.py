import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence

from config.config import Config


class LSTMFeedforwardCombination(nn.Module):
    def __init__(self, cfg: Config, lstm_input_dim, ff_input_dim):
        super().__init__()
        lstm_hidden_dim = cfg.model.lstm.lstm_hidden_dim
        ff_hidden_dims = cfg.model.lstm.ff_hidden_dims
        dropout = cfg.model.lstm.dropout
        self.name = "LSTM"
        self.bidirectional = cfg.model.lstm.bidirectional
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim, bidirectional=cfg.model.lstm.bidirectional,
                            num_layers=cfg.model.lstm.num_lstm_layers, batch_first=True, dropout=dropout)
        self.lstm_dropout = nn.Dropout(dropout)
        lstm_output_dim = 2 * lstm_hidden_dim if self.bidirectional else lstm_hidden_dim
        self.ln = nn.LayerNorm(lstm_output_dim)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden_list = nn.ModuleList()
        self.fc_hidden_list.append(nn.Linear(ff_input_dim, ff_hidden_dims[0]))
        for i in range(len(ff_hidden_dims) - 1):
            self.fc_hidden_list.append(nn.Linear(ff_hidden_dims[i], ff_hidden_dims[i + 1]))

        self.final = nn.Sequential(
            nn.Linear(lstm_output_dim + ff_hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, time_features, padded_routes, lengths):
        if torch.onnx.is_in_onnx_export():
            packed = padded_routes
        else:
            packed = pack_padded_sequence(padded_routes, lengths, batch_first=True, enforce_sorted=False)

        _, (hn, _) = self.lstm(packed)
        if self.bidirectional:
            hn = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            hn = hn[-1]
        hn = self.ln(hn)
        hn = self.lstm_dropout(hn)

        tf = time_features
        for hidden_layer in self.fc_hidden_list:
            tf = hidden_layer(tf)
            tf = self.act(tf)
            tf = self.dropout(tf)

        combined = torch.cat((hn, tf), dim=1)
        output = self.final(combined)
        return output