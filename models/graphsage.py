import torch.nn as nn
from torch_geometric.nn import SAGEConv
from typing import Literal


class GraphSAGE(nn.Module):
    def __init__(self, in_channels=128, out_channels=40, hidden_channels=256, num_layers=3, dropout=0.5, 
                 aggr:Literal['mean','lstm','max']='mean'):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.conv_layers.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.bn_layers.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers-2):
            self.conv_layers.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bn_layers.append(nn.BatchNorm1d(hidden_channels))
        self.classifier = SAGEConv(hidden_channels, out_channels, aggr=aggr)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.relu(x)
            x = self.dropout(x)
        logits = self.classifier(x, edge_index)
        return logits