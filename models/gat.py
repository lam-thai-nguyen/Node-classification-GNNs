import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, in_channels=128, out_channels=40, hidden_channels=128, num_layers=3, dropout=0.5, skip=False, mlp=False):
        super().__init__()

        self.mlp = mlp
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        # pre-MLP and post-MLP
        if mlp:
            self.pre_mlp = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(inplace=True), nn.Dropout(dropout))
            self.post_mlp = nn.Sequential(
                nn.Linear(out_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, out_channels),
            )
            gnn_in_channels = hidden_channels
        else:
            self.pre_mlp = None
            self.post_mlp = None
            gnn_in_channels = in_channels

        # GNN stack
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.conv_layers.append(GATConv(gnn_in_channels, hidden_channels, heads=1, concat=True, dropout=dropout))
        self.bn_layers.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers-2):
            self.conv_layers.append(GATConv(hidden_channels, hidden_channels, heads=1, concat=True, dropout=dropout))
            self.bn_layers.append(nn.BatchNorm1d(hidden_channels))
        self.conv_layers.append(GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout))
        self.elu = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        # Skip connection projection
        self.residual_proj = nn.Linear(gnn_in_channels, hidden_channels) if skip else None
    
    def forward(self, x, edge_index, return_layer=None):
        if self.pre_mlp is not None:
            x = self.pre_mlp(x)

        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            identity = x
            if self.residual_proj is not None and i == 0:
                identity = self.residual_proj(x)
            x = conv(x, edge_index)
            x = bn(x)
            if self.residual_proj is not None:
                x = x + identity
            x = self.elu(x)
            x = self.dropout(x)
            if return_layer == i:
                return x
            
        logits = self.conv_layers[-1](x, edge_index)
        if self.post_mlp is not None:
            logits = self.post_mlp(logits)

        return logits
    