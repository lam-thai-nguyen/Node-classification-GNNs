import yaml
import torch
from ogb.nodeproppred import Evaluator
from models import GCN
from utils import num_params, load_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("using", DEVICE)

@torch.no_grad()
def evaluate(model, graph, split_idx):
    model.eval()
    model = model.to(DEVICE)
    x, y, edge_index = graph.x.to(DEVICE), graph.y.to(DEVICE), graph.edge_index.to(DEVICE)

    train_idx = split_idx["train"]
    valid_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    out = model(x, edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)
    y_pred = y_pred.detach().cpu()

    evaluator = Evaluator(name='ogbn-arxiv')

    train_acc = evaluator.eval({
        'y_true': y[train_idx],
        'y_pred': y_pred[train_idx],
    })['acc']

    valid_acc = evaluator.eval({
        'y_true': y[valid_idx],
        'y_pred': y_pred[valid_idx],
    })['acc']

    test_acc = evaluator.eval({
        'y_true': y[test_idx],
        'y_pred': y_pred[test_idx],
    })['acc']

    return train_acc, valid_acc, test_acc


if __name__ == "__main__":
    graph, split_idx = load_dataset()
    
    # Select a config path only, other arguments are automatically derived
    config_path = "configs/gcn_baseline.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    checkpoint = torch.load(cfg['best_checkpoint_dir'], map_location=DEVICE)
    gcn = GCN(cfg['in_channels'], cfg['out_channels'], cfg['d_model'], cfg['layers'], cfg['dropout'])
    print("num params:", num_params(gcn))
    gcn.load_state_dict(checkpoint["model_state_dict"])

    train_acc, valid_acc, test_acc = evaluate(gcn, graph, split_idx)
    print(f"train accuracy {train_acc*100:.2f}%\t val accuracy {valid_acc*100:.2f}%\t test accuracy {test_acc*100:.2f}%\t")
