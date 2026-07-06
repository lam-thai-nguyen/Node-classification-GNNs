import torch
from sklearn.preprocessing import StandardScaler
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14


def num_params(model):
    params = sum(p.numel() for p in model.parameters())
    return params

def plot_history(train_history, val_history, acc_history):
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
    Args:
        gnn (bool): Whether you are loading for a GNN model or not. Default to False.
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

def tsne(model, return_layer, data, subsample=10000):
    """
    Args:
        model: your model
        return_layer: which layer should return embeddings. starting from 0.
        data: your graph
    """
    import os
    os.environ["LOKY_MAX_CPU_COUNT"] = "6"
    
    model = model.to("cpu")
    model.eval()
    with torch.no_grad():
        out = model(data.x, return_layer).numpy()
        print("node embedding shape:", out.shape)
    
    np.random.seed(42)
    idx = np.random.choice(data.num_nodes, subsample, replace=False)
    tsne = TSNE(n_components=2, init='pca', random_state=42).fit_transform(out[idx])

    plt.figure(figsize=(7,6))
    sc = plt.scatter(tsne[:,0], tsne[:,1], c=data.y[idx], cmap='tab20', s=5, alpha=0.7)
    plt.colorbar(sc, label='Class')
    plt.savefig(f"tsne_{return_layer}.jpg", dpi=300, bbox_inches="tight")
    print(f"✅ Saved tsne_{return_layer}.jpg")
