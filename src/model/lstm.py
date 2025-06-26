import torch.nn as nn
import torch

class LSTMFeedforwardCombination(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, time_input_dim, ff_hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim, num_layers=1, batch_first=True)
        self.time_ff = nn.Sequential(
            nn.Linear(time_input_dim, ff_hidden_dim),
            nn.ReLU()
        )
        self.final = nn.Sequential(
            nn.Linear(lstm_input_dim + ff_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data_dict: dict):
        route_sequence = data_dict['route_sequence']
        time_features = data_dict['time_features']
        _, (hn, _) = self.lstm(route_sequence)
        hn = hn.squeeze(0)

        time_emb = self.time_fc(time_features)

        combined = torch.cat((hn, time_emb), dim=1)
        output = self.final_fc(combined)
        return output