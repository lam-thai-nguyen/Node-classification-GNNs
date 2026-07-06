import yaml
import torch
from ogb.nodeproppred import Evaluator
from models import MLP
from utils import num_params, plot_history, load_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("using", DEVICE)

def train(mlp, x, y, split_idx, optimizer, loss_fn, epochs, best_checkpoint_dir, checkpoint_dir):
    """
    Performs batch optimization for MLP

    Args:
        mlp (nn.Module): model
        x (tensor): examples
        y (tensor): targets
        split_idx (tensor): split_idx
        optimizer: optimizer
        loss_fn (nn.Module): loss function
        epochs (int): number of epochs
        best_checkpoint_dir (str): where to save best checkpoint
        checkpoint_dir (str): where to save last checkpoint

    Returns:
        None
    """
    mlp = mlp.to(DEVICE)
    train_history = []
    val_history = []
    acc_history = []
    best_val_loss = float("inf")
    
    for i in range(epochs):
        mlp.train()
        train_loss = 0.0

        optimizer.zero_grad()
        examples = x[split_idx["train"]].to(DEVICE)
        targets = y[split_idx["train"]].squeeze(-1).to(DEVICE)
        logits = mlp(examples)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        val_loss = eval(mlp, x, y, split_idx["valid"], loss_fn)
        val_acc = accuracy(mlp, x, y, split_idx["valid"])
        train_history.append(train_loss)
        val_history.append(val_loss)
        acc_history.append(val_acc)

        if i % 10 == 0:
            print(f"Epoch {i:<6} train loss {train_loss:<10.4f} val loss {val_loss:<10.4f} val accuracy {val_acc:<10.4f}")

        # Save best checkpoint based on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": i,
                "model_state_dict": mlp.state_dict(),
                "train_history": train_history,
                "val_history": val_history,
                "acc_history": acc_history,
                "best_val_loss": best_val_loss,
            }, best_checkpoint_dir)
    
    # Save last checkpoint
    torch.save({
        "epoch": epochs,
        "model_state_dict": mlp.state_dict(),
        "train_history": train_history,
        "val_history": val_history,
        "acc_history": acc_history,
    }, checkpoint_dir)

@torch.no_grad()
def eval(mlp, x, y, valid_idx, loss_fn):
    """
    Computes the cross-validation loss

    Args:
        mlp (nn.Module): model
        x (tensor): examples
        y (tensor): targets
        valid_idx (tensor): valid_idx
        loss_fn (nn.Module): loss function
    
    Returns:
        loss (float): val loss
    """
    mlp = mlp.to(DEVICE)
    mlp.eval()
    
    examples = x[valid_idx].to(DEVICE)
    targets = y[valid_idx].squeeze(-1).to(DEVICE)
    logits = mlp(examples)
    loss = loss_fn(logits, targets)

    return loss.item()

@torch.no_grad()
def accuracy(mlp, x, y, idx):
    """
    Accuracy using OGB Evaluator

    Args:
        mlp (nn.Module): model
        x (tensor): examples
        y (tensor): targets
        idx (tensor): expects train_idx, valid_idx, or test_idx

    Returns:
        acc (float): accuracy
    """
    mlp = mlp.to(DEVICE)
    mlp.eval()

    examples = x[idx].to(DEVICE)
    logits = mlp(examples)
    y_pred = torch.argmax(logits, dim=-1).unsqueeze(-1).detach().cpu()

    evaluator = Evaluator(name='ogbn-arxiv')
    acc = evaluator.eval({
        "y_true": y[idx],
        "y_pred": y_pred
    })

    return acc['acc']

if __name__ == "__main__":
    data, split_idx = load_dataset()

    # Select a config path only, other arguments are automatically derived
    config_path = "configs/baseline/mlp.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mlp = MLP(cfg['in_channels'], cfg['out_channels'], cfg['hidden_channels'], cfg['layers'], cfg['dropout'])
    print("num params:", num_params(mlp))
    optimizer = torch.optim.Adam(mlp.parameters(), lr=cfg['lr'])
    loss_fn = torch.nn.CrossEntropyLoss()
    train(mlp, data.x, data.y, split_idx, optimizer, loss_fn, cfg['epochs'], cfg['best_checkpoint_dir'], cfg['checkpoint_dir'])

    checkpoint = torch.load(cfg['best_checkpoint_dir'], map_location=DEVICE)
    train_history = checkpoint["train_history"]
    val_history = checkpoint["val_history"]
    acc_history = checkpoint["acc_history"]
    plot_history(train_history, val_history, acc_history)
