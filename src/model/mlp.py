import torch.nn as nn

from config.config import Config


class MLP(nn.Module):
    def __init__(self, cfg: Config, input_dim: int):
        super().__init__()
        hidden_dims = cfg.model.mlp.hidden_dims
        output_dim = cfg.model.output_dim
        self.name = "MLP"

        self.fc_in = nn.Linear(input_dim, hidden_dims[0])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.model.mlp.dropout)

        self.fc_hidden_list = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            for i in range(len(hidden_dims) - 1)
        )
        self.fc_out = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = self.act(self.fc_in(x))
        for fc_hidden in self.fc_hidden_list:
            x = self.dropout(self.act(fc_hidden(x)))
        return self.fc_out(x)
