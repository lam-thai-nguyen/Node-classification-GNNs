"""Table 4 - ablation study on OGBN-ARXIV (fixed depth, multiple runs)."""

import os
import sys

_RUN_LOCAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUN_LOCAL_DIR not in sys.path:
    sys.path.insert(0, _RUN_LOCAL_DIR)

import time

import torch

import common
import config
from learning import train, evaluate
from reporting import save_results_table, summarize_runs

CHECKPOINT_SUBDIR = "ablation"
COLUMNS = ["Family", "Model", "Params", "Train acc (%)", "Valid acc (%)", "Test acc (%)"]


def _run_family(family_label, variants, cfg, data, split_idx, in_channels,
                loss_fn, ckpt_dir):
    from utils import num_params

    rows = []
    total_time = 0.0
    for name, ctor in variants.items():
        start = time.perf_counter()
        train_accs, valid_accs, test_accs, n_params = [], [], [], None

        for run_id in range(1, cfg["n_runs"] + 1):
            common.set_seed(run_id)
            model = ctor(in_channels, config.OUT_CHANNELS,
                         cfg["hidden_channels"], cfg["num_layers"], cfg["dropout"])
            n_params = num_params(model)
            print(f"RUN {run_id} | {name:<10} params={n_params:,}")

            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            best_path = os.path.join(ckpt_dir, f"{name.lower()}_best_{run_id}.pt")
            last_path = os.path.join(ckpt_dir, f"{name.lower()}_last_{run_id}.pt")
            train(model, data, split_idx, optimizer, loss_fn, cfg["epochs"],
                  best_path, last_path)

            ckpt = torch.load(best_path, map_location=common.DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            tr, va, te = evaluate(model, data, split_idx)
            train_accs.append(tr)
            valid_accs.append(va)
            test_accs.append(te)

        elapsed = time.perf_counter() - start
        total_time += elapsed
        summary = summarize_runs(train_accs, valid_accs, test_accs)
        rows.append({
            "Family": family_label,
            "Model": name,
            "Params": f"{n_params:,}",
            "Train acc (%)": summary["train"],
            "Valid acc (%)": summary["valid"],
            "Test acc (%)": summary["test"],
        })
        print(f"{name}: test {summary['test']}  ({elapsed / 60:.2f} mins)\n")

    return rows, total_time


def run(results_dir=None):
    results_dir = results_dir or common.RESULTS_DIR
    cfg = config.ABLATION

    ckpt_dir = common.ensure_dir(os.path.join(results_dir, "checkpoints", CHECKPOINT_SUBDIR))
    table_dir = common.ensure_dir(os.path.join(results_dir, "tables"))

    data, split_idx = common.get_dataset(gnn=True)
    in_channels = data.x.shape[1]
    loss_fn = torch.nn.CrossEntropyLoss()

    gcn_rows, t1 = _run_family("GCN", config.GCN_VARIANTS, cfg, data, split_idx,
                               in_channels, loss_fn, ckpt_dir)
    sage_rows, t2 = _run_family("GraphSAGE", config.SAGE_VARIANTS, cfg, data, split_idx,
                                in_channels, loss_fn, ckpt_dir)
    rows = gcn_rows + sage_rows

    save_results_table(
        rows, COLUMNS,
        csv_path=os.path.join(table_dir, "table4_ablation.csv"),
        md_path=os.path.join(table_dir, "table4_ablation.md"),
        title=f"Table 4 - Ablation study (num_layers={cfg['num_layers']}) on OGBN-ARXIV",
    )
    print(f"ALL DONE (ablation). Total {(t1 + t2) / 60:.2f} mins")
    return rows


if __name__ == "__main__":
    run()
