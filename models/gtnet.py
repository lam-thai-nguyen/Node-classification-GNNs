import torch.nn as nn
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import MessagePassing


class GTCNConv(MessagePassing):
    """
    The GTCNConv layer is a custom PyG MessagePassing-inherited class.
    This layer doesn't introduce any weights. For hidden feature of node v (h_v), 
    GTCNConv aggregating its neighbors' output hidden features from the previous propagation layer and its own initial feature.
    Template used: https://pytorch-geometric.readthedocs.io/en/2.6.0/tutorial/create_gnn.html#implementing-the-gcn-layer
    """
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, H, Z, edge_index):
        """
        Args:
            H: previous node embeddings (N,F)
            Z: MLP(x) where x is initial embedding (N,F)
            edge_index: edge list (2,E)

        Returns:
            (N,F): new embedding - equation (9) from the paper
        """
        # add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=H.size(0))  # (2,E) -> (2,E+N)

        # compute normalization
        row, col = edge_index  # row is source nodes, col is dst nodes, both are (E+N,)
        deg = degree(col, H.size(0), dtype=H.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # (E+N,)

        # get A1 and A2 in equation (9)
        self_loop_mask = row == col  # (E+N,)
        A2 = norm[self_loop_mask]  # (N,) -- A2[i] = A[i,i]
        A1 = norm[~self_loop_mask]  # (E,)
        A1_edge_index = edge_index[:, ~self_loop_mask]  # (2,E) -- the original edge index without added self-loops
        
        # propagating messages
        out = self.propagate(A1_edge_index, x=H, norm=A1)  # (N,F)
        out = out + A2.view(-1, 1) * Z  # (N,F) + (N,1)*(N,F)

        return out

    def message(self, x_j, norm):
        """
        Args:
            x_j: features of source node for each edge (E,F)
            norm: normalization (E,)

        Returns:
            MSG (E,F): one message for each edge
        """
        return norm.view(-1, 1) * x_j  # (E,1) * (E,F) (element-wise mul.)

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
