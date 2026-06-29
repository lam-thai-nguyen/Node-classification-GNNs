import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_channels=128, out_channels=40, hidden_channels=256, num_layers=3, dropout=0.5):
        super().__init__()

        self.linear_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.linear_layers.append(nn.Linear(in_channels, hidden_channels, bias=True))
        self.bn_layers.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers-2):
            self.linear_layers.append(nn.Linear(hidden_channels, hidden_channels, bias=True))
            self.bn_layers.append(nn.BatchNorm1d(hidden_channels))
        self.classifier = nn.Linear(hidden_channels, out_channels, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for linear_layer, bn_layer in zip(self.linear_layers, self.bn_layers):
            x = linear_layer(x)
            x = bn_layer(x)
            x = self.relu(x)
            x = self.dropout(x)
        logits = self.classifier(x)
        return logits
    