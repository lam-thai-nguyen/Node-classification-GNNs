import torch.nn as nn
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import MessagePassing


class GTCNConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, h, z, edge_index):
        """
        Args:
            h: previous node embeddings (N,F)
            z: MLP(x) where x is initial embedding (N,F)
            edge_index: edge list (2,E)

        Returns:
            (N.F): new embedding - equation (8) from the paper
        """
        edge_index, _ = add_self_loops(edge_index, num_nodes=h.size(0))  # (2,E) -> (2,E+N)

        row, col = edge_index  # row is source nodes (E+N,), col is dst nodes (E+N,)
        deg = degree(row, h.size(0), dtype=h.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # look at eq (8) from paper
        # currently, A_uu is non-zero, so a part of the current node's previous embedding is used (which is unwanted)
        # we set A_uu to zero first, so the current node's previous embedding is not used
        # later, we set A_uu back to use z for aggregation
        self_loop_mask = row == col
        norm_h = norm.clone()
        norm_h[self_loop_mask] = 0.0

        out = self.propagate(edge_index, x=h, norm=norm_h)

        # we set A_uu back to use z for aggregation
        A_uu = norm[self_loop_mask]  # assumes no self-loops in the original edge_index (fragile but usable for OGBN-ARXIV)
        out = out + A_uu.unsqueeze(-1) * z

        return out

    def message(self, x_j, norm):
        """
        Args:
            x_j: features of source node for each edge (E+N,F)
            norm: the normalization weights (norm_h) (E+N,)

        Returns:
            MSG (E+N,F): one message for each edge
        """
        return norm.view(-1, 1) * x_j

class GTCN(nn.Module):
    def __init__(self, in_channels=128, out_channels=40, hidden_channels=256, num_layers=5, dropout=0.2):
        super().__init__()
        self.mlp1 = nn.Linear(in_channels, hidden_channels)
        self.mlp2 = nn.Linear(hidden_channels, hidden_channels)
        self.convs = nn.ModuleList([GTCNConv() for _ in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        z = self.relu(self.mlp1(self.dropout(x)))
        z = self.mlp2(self.dropout(z))
        h = z
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, z, edge_index)
            h = bn(h)

        h = self.dropout(self.relu(h))
        logits = self.classifier(h)
        return logits
