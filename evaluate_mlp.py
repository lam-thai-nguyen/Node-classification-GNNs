import yaml
import torch
from ogb.nodeproppred import Evaluator
from models import MLP
from utils import num_params, load_dataset, tsne


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("using", DEVICE)

@torch.no_grad()
def evaluate(model, x, y, split_idx):
    model.eval()
    model = model.to(DEVICE)
    x = x.to(DEVICE)

    out = model(x)
    y_pred = out.argmax(dim=-1)
    y_pred = y_pred.unsqueeze(-1).detach().cpu()

    evaluator = Evaluator(name='ogbn-arxiv')

    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']

    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']

    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


if __name__ == "__main__":
    data, split_idx = load_dataset()
    
    # Select a config path only, other arguments are automatically derived
    config_path = "configs/baseline/mlp.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    checkpoint = torch.load(cfg['best_checkpoint_dir'], map_location=DEVICE)
    mlp = MLP(cfg['in_channels'], cfg['out_channels'], cfg['hidden_channels'], cfg['layers'], cfg['dropout'])
    print("num params:", num_params(mlp))
    mlp.load_state_dict(checkpoint["model_state_dict"])

    train_acc, valid_acc, test_acc = evaluate(mlp, data.x, data.y, split_idx)
    print(f"train accuracy {train_acc*100:.2f}%\t val accuracy {valid_acc*100:.2f}%\t test accuracy {test_acc*100:.2f}%\t")

    tsne(mlp, 1, data, 100)
