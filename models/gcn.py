import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_channels=128, out_channels=40, hidden_channels=256, num_layers=3, dropout=0.5, skip=False):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bn_layers.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers-2):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bn_layers.append(nn.BatchNorm1d(hidden_channels))
        self.classifier = GCNConv(hidden_channels, out_channels, cached=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.residual_proj = nn.Linear(in_channels, hidden_channels) if skip else None

    def forward(self, x, edge_index):
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            identity = x
            if self.residual_proj is not None and i == 0:
                identity = self.residual_proj(x)
            x = conv(x, edge_index)
            x = bn(x)
            if self.residual_proj is not None:
                x = x + identity
            x = self.relu(x)
            x = self.dropout(x)
        logits = self.classifier(x, edge_index)
        return logits