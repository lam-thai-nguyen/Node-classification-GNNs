import torch
from sklearn.preprocessing import StandardScaler
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14


def num_params(model):
    """
    Calculate the number of parameters of a model

    Args:
        model (nn.Module): the model

    Returns:
        params (int): number of parameters
    """
    params = sum(p.numel() for p in model.parameters())
    return params

def plot_history(train_history, val_history, acc_history):
    """
    Plots and saves two figures: train-valid loss and valid accuracy

    Args:
        train_history, val_history, acc_history (list): these are outputs from your train() function
    
    Returns:
        None
    """
    epochs = range(1, len(train_history) + 1)

    # Figure 1: train vs val loss
    plt.figure(figsize=(5, 5))
    plt.plot(epochs, train_history, label="Train Loss")
    plt.plot(epochs, val_history, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("loss.jpg", dpi=150, bbox_inches="tight")
    print("✅ Saved loss.jpg")

    # Figure 2: val accuracy
    plt.figure(figsize=(5, 5))
    plt.plot(epochs, acc_history, label="Val Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("acc.jpg", dpi=150, bbox_inches="tight")
    print("✅ Saved acc.jpg")

def load_dataset(gnn=False):
    """
    Loads OGBN-ARXIV

    Args:
        gnn (bool): Whether you are loading for a GNN model or not. Default to False.

    Returns:
        data, split_idx: the graph and its split indices
    """
    dataset = PygNodePropPredDataset(name='ogbn-arxiv') 

    split_idx = dataset.get_idx_split()
    data = dataset[0]

    scaler = StandardScaler()
    data.x = torch.from_numpy(scaler.fit_transform(data.x.numpy()))

    if gnn:
        data.edge_index = to_undirected(data.edge_index)
        # skip removing self-loops because a citation network can't have self-loops

    return data, split_idx
