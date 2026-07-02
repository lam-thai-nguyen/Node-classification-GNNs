import yaml
import torch
from ogb.nodeproppred import Evaluator
from models import GCN, GraphSAGE, GAT
from utils import num_params, plot_history, load_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("using", DEVICE)

def train(gnn, graph, split_idx, optimizer, loss_fn, epochs, best_checkpoint_dir, checkpoint_dir):
    """
    Performs batch optimization for GNN

    Args:
        gnn (nn.Module): model
        graph: your graph
        split_idx (tensor): split_idx
        optimizer: optimizer
        loss_fn (nn.Module): loss function
        epochs (int): number of epochs
        best_checkpoint_dir (str): where to save best checkpoint
        checkpoint_dir (str): where to save last checkpoint

    Returns:
        None
    """
    gnn = gnn.to(DEVICE)
    train_history = []
    val_history = []
    acc_history = []
    best_val_loss = float("inf")
    
    train_idx = split_idx["train"]
    valid_idx = split_idx["valid"]
    
    for i in range(epochs):
        gnn.train()
        train_loss = 0.0

        # train
        optimizer.zero_grad()
        x, y, edge_index = graph.x.to(DEVICE), graph.y.to(DEVICE), graph.edge_index.to(DEVICE)
        logits = gnn(x, edge_index)
        logits_train = logits[train_idx]
        train_loss = loss_fn(logits_train, y.squeeze(-1)[train_idx])
        train_loss.backward()
        optimizer.step()
        train_history.append(train_loss.item())

        # val
        logits_valid = logits[valid_idx]
        val_loss = loss_fn(logits_valid, y.squeeze(-1)[valid_idx])
        val_history.append(val_loss.item())

        # accuracy
        evaluator = Evaluator(name='ogbn-arxiv')
        y_true = y[valid_idx]
        y_pred = torch.argmax(logits_valid, dim=-1, keepdim=True).detach().cpu()
        val_acc = evaluator.eval({
            "y_true": y_true,
            "y_pred": y_pred
        })['acc']
        acc_history.append(val_acc)
        

        if i % 10 == 0:
            print(f"Epoch {i:<6} train loss {train_loss:<10.4f} val loss {val_loss:<10.4f} val accuracy {val_acc:<10.4f}")

        # Save best checkpoint based on val_loss
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save({
                "epoch": i,
                "model_state_dict": gnn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_history": train_history,
                "val_history": val_history,
                "acc_history": acc_history,
                "best_val_loss": best_val_loss,
            }, best_checkpoint_dir)
    
    # Save last checkpoint
    torch.save({
        "epoch": epochs,
        "model_state_dict": gnn.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_history": train_history,
        "val_history": val_history,
        "acc_history": acc_history,
    }, checkpoint_dir)

def train_gcn():
    data, split_idx = load_dataset(gnn=True)

    # Select a config path only, other arguments are automatically derived
    config_path = "configs/gcn_baseline.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    gnn = GCN(cfg['in_channels'], cfg['out_channels'], cfg['d_model'], cfg['layers'], cfg['dropout'])
    print("num params:", num_params(gnn))
    optimizer = torch.optim.Adam(gnn.parameters(), lr=cfg['lr'])
    loss_fn = torch.nn.CrossEntropyLoss()
    train(gnn, data, split_idx, optimizer, loss_fn, cfg['epochs'], cfg['best_checkpoint_dir'], cfg['checkpoint_dir'])

    checkpoint = torch.load(cfg['best_checkpoint_dir'], map_location=DEVICE)
    train_history = checkpoint["train_history"]
    val_history = checkpoint["val_history"]
    acc_history = checkpoint["acc_history"]
    plot_history(train_history, val_history, acc_history)

def train_sage():
    data, split_idx = load_dataset(gnn=True)

    # Select a config path only, other arguments are automatically derived
    config_path = "configs/sage_baseline.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    gnn = GraphSAGE(cfg['in_channels'], cfg['out_channels'], cfg['d_model'], cfg['layers'], cfg['dropout'], cfg['aggr'])
    print("num params:", num_params(gnn))
    optimizer = torch.optim.Adam(gnn.parameters(), lr=cfg['lr'])
    loss_fn = torch.nn.CrossEntropyLoss()
    train(gnn, data, split_idx, optimizer, loss_fn, cfg['epochs'], cfg['best_checkpoint_dir'], cfg['checkpoint_dir'])

    checkpoint = torch.load(cfg['best_checkpoint_dir'], map_location=DEVICE)
    train_history = checkpoint["train_history"]
    val_history = checkpoint["val_history"]
    acc_history = checkpoint["acc_history"]
    plot_history(train_history, val_history, acc_history)

def train_gat():
    data, split_idx = load_dataset(gnn=True)

    # Select a config path only, other arguments are automatically derived
    config_path = "configs/gat_baseline.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    gat = GAT(cfg['in_channels'], cfg['out_channels'], cfg['d_model'], cfg['heads'], cfg['layers'], cfg['dropout'])
    print("num params:", num_params(gat))
    optimizer = torch.optim.Adam(gat.parameters(), lr=cfg['lr'])
    loss_fn = torch.nn.CrossEntropyLoss()
    train(gat, data, split_idx, optimizer, loss_fn, cfg['epochs'], cfg['best_checkpoint_dir'], cfg['checkpoint_dir'])

    checkpoint = torch.load(cfg['best_checkpoint_dir'], map_location=DEVICE)
    train_history = checkpoint["train_history"]
    val_history = checkpoint["val_history"]
    acc_history = checkpoint["acc_history"]
    plot_history(train_history, val_history, acc_history)

if __name__ == "__main__":
    # train_gcn()
    # train_sage()
    train_gat()
