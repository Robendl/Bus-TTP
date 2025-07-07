import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        self.name = "MLP"
        self.fc_in = nn.Linear(input_dim, hidden_dims[0])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc_hidden_list = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.fc_hidden_list.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.fc_out = nn.Linear(hidden_dims[-1], output_dim)


    def forward(self, x):
        x = self.fc_in(x)
        x = self.relu(x)
        for fc_hidden in self.fc_hidden_list:
            x = fc_hidden(x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.fc_out(x)
        return x
