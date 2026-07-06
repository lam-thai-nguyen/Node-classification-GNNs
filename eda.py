import collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import load_dataset
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14


def degree_distribution(edge_index):
    """
    Plots and saves the log-log degree distribution

    Args:
        edge_index (np.ndarray): The edge list of the graph

    Returns:
        None
    """
    # deg[i] is the (in degree + out degree) of node i
    deg = np.bincount(np.concatenate([edge_index[0], edge_index[1]]))

    # count the frequency of degree values
    deg_count = collections.Counter(deg)
    degs, counts = zip(*sorted(deg_count.items()))

    plt.figure(figsize=(6,5))
    plt.loglog(degs, counts, 'o', markersize=3, alpha=0.5)
    plt.xlabel('Degree (log)')
    plt.ylabel('Count (log)')
    plt.savefig("degree_distribution.jpg", dpi=150, bbox_inches="tight")
    print("✅ Saved degree_distribution.jpg")

    print(f"Mean degree: {deg.mean():.2f}, Max degree: {deg.max()}, Median: {np.median(deg)}")

def label_distribution(y):
    """
    Plots and saves a bar plot showing the frequency of each class

    Args:
        y (np.ndarray): The label array for every example

    Returns:
        None
    """
    class_counts = np.bincount(y)
    plt.figure(figsize=(10,4))
    plt.bar(np.arange(40), class_counts)
    plt.xlabel('Class')
    plt.ylabel('Number of nodes')
    plt.savefig("label_distribution.jpg", dpi=150, bbox_inches="tight")
    print("✅ Saved label_distribution.jpg")

    print(f"Largest class: {class_counts.max()} nodes, Smallest: {class_counts.min()} nodes")

def temporal_structure(split_idx):
    """
    Plots and saves the frequency of each node in each year

    Args:
        split_idx (dict): Refer to https://ogb.stanford.edu/docs/nodeprop/#pyg
    
    Returns:
        None
    """
    years, ycounts = np.unique(node_year, return_counts=True)
    plt.figure(figsize=(8,4))
    plt.bar(years, ycounts)
    plt.xlabel('Year')
    plt.ylabel('Number of papers')
    plt.axvline(x=node_year[split_idx['train']].max(), color='g', linestyle='--', label='train cutoff')
    plt.axvline(x=node_year[split_idx['valid']].max(), color='orange', linestyle='--', label='valid cutoff')
    plt.legend()
    plt.savefig("temporal_structure.jpg", dpi=150, bbox_inches="tight")
    print("✅ Saved temporal_structure.jpg")

def tsne(subsample, num_nodes):
    """
    Use t-SNE to map nodes from high-dimensional space to 2. Plots and saves the 2d nodes to see how separable classes are based on node features.

    Args:
        subsample (int): number of nodes to sample from all nodes
        num_nodes (int): number of nodes

    Returns:
        None 
    """
    import os
    os.environ["LOKY_MAX_CPU_COUNT"] = "6"

    assert subsample <= num_nodes, "choose subsample <= num_nodes"

    np.random.seed(42)
    idx = np.random.choice(num_nodes, subsample, replace=False)
    tsne = TSNE(n_components=2, init='pca', random_state=42).fit_transform(x[idx])

    plt.figure(figsize=(7,6))
    sc = plt.scatter(tsne[:,0], tsne[:,1], c=y[idx], cmap='tab20', s=5, alpha=0.7)
    plt.colorbar(sc, label='Class')
    plt.savefig("tsne.jpg", dpi=300, bbox_inches="tight")
    print("✅ Saved tsne.jpg")

def homophily(edge_index, y):
    """
    Print the edge homophily. ie. fraction of edges connecting same-class nodes

    Args:
        edge_index (np.ndarray): The edge list of the graph
        y (np.ndarray): The label array for every example

    Returns:
        None
    """
    src, dst = edge_index[0], edge_index[1]
    same_label = (y[src] == y[dst]).mean()
    print(f"Edge homophily: {same_label:.3f}")

if __name__ == "__main__":
    data, split_idx = load_dataset()

    edge_index = data.edge_index.numpy()
    x = data.x
    y = data.y.flatten()
    node_year = data.node_year.numpy().flatten()
    num_nodes = data.num_nodes

    # EDA
    degree_distribution(edge_index)
    label_distribution(y)
    temporal_structure(split_idx)
    tsne(subsample=10000, num_nodes=num_nodes)
    homophily(edge_index, y)