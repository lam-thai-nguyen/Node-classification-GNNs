import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, in_channels=128, out_channels=40, hidden_channels=128, num_layers=3, dropout=0.5):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.conv_layers.append(GATConv(in_channels, hidden_channels, heads=1, concat=True, dropout=dropout))
        self.bn_layers.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers-2):
            self.conv_layers.append(GATConv(hidden_channels, hidden_channels, heads=1, concat=True, dropout=dropout))
            self.bn_layers.append(nn.BatchNorm1d(hidden_channels))
        self.classifier = GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout)
        self.elu = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, edge_index):
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.elu(x)
            x = self.dropout(x)
        logits = self.classifier(x, edge_index)
        return logits
    