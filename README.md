# Combating Over-smoothing in GNNs: Skip Connections vs. Layer Aggregation on OGBN-ARXIV

This is the code for *Combating Over-smoothing in GNNs: Skip Connections vs. Layer Aggregation on OGBN-ARXIV*, a project for *IT5429E - Graph analytics for big data* (Master's course @ HUST).

> All experiments are conducted in a Kaggle notebook environment equipped with an NVIDIA Tesla T4 GPU (16GB VRAM), 13GB RAM, and 2 CPU cores, using PyTorch and PyTorch Geometric for implementation.

The mentioned Kaggle notebook can be found [[here]](https://www.kaggle.com/code/thaimeuu/it5429e-workspace), which uses Kaggle utility scripts that are based on this repository.

| URL | Description | Based on |
|---|---|---|
https://www.kaggle.com/code/thaimeuu/it5429e-workspace | Experiments (outputs, which are used in the reported, are saved in this notebook) | Independent |
| https://www.kaggle.com/code/thaimeuu/graph-ml-it5429e-models | All models | [models/](models/) |
| https://www.kaggle.com/code/thaimeuu/graph-ml-it5429e-learning | Training and evaluation functions | Independent |
| https://www.kaggle.com/code/thaimeuu/graph-ml-it5429e-utils | Utils functions | [utils.py](utils.py) |

The figures for OGBN-ARXIV in the report can be found here: [images/](images/)

The figures for the experiments can be found [in this Kaggle notebook](https://www.kaggle.com/code/thaimeuu/it5429e-workspace)

The checkpoints for the experiments can be found here: [checkpoints/](checkpoints/)

| File | Description | Section in report |
|---|---|---|
| [checkpoints/transductive/](checkpoints/transductive/) | The main result | Section 4.1, Table 3 |
| [checkpoints/oversmoothing/](checkpoints/oversmoothing/) | Over-smoothing analysis | Section 4.2, Figures 5, 6, 7 |
| [checkpoints/ablation/](checkpoints/ablation/) | Ablation study | Section 4.3, Table 4 |
| [checkpoints/appendix/](checkpoints/appendix/) | Over-smoothing comparison GTCN vs GCN variants | Appendix A, Figures 8, 9 |