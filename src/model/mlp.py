import torch.nn as nn

from config.config import Config


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        input_dim = len(cfg.dataset.time_feature_names) + len(cfg.dataset.route_feature_names)
        hidden_dims = cfg.model.mlp.hidden_dims
        output_dim = cfg.model.output_dim
        super(MLP, self).__init__()
        self.name = "MLP"
        self.fc_in = nn.Linear(input_dim, hidden_dims[0])
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(cfg.model.mlp.dropout)
        self.fc_hidden_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.fc_hidden_list.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.bn_list.append(nn.BatchNorm1d(hidden_dims[i + 1]))
        self.fc_out = nn.Linear(hidden_dims[-1], output_dim)


    def forward(self, x):
        x = self.fc_in(x)
        x = self.relu(x)
        for fc_hidden, batch_norm in zip(self.fc_hidden_list, self.bn_list):
            x = fc_hidden(x)
            # x = batch_norm(x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.fc_out(x)
        return x
