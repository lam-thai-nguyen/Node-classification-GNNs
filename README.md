# Combating Over-smoothing in GNNs: Skip Connections vs. Layer Aggregation on OGBN-ARXIV

This is the code for *Combating Over-smoothing in GNNs: Skip Connections vs. Layer Aggregation on OGBN-ARXIV*, a project for *IT5429E - Graph analytics for big data* (Master's course @ HUST).

> All experiments are conducted in a Kaggle notebook environment equipped with an NVIDIA Tesla T4 GPU (16GB VRAM), 13GB RAM, and 2 CPU cores, using PyTorch and PyTorch Geometric for implementation.

The mentioned Kaggle notebook can be found [[here]](https://www.kaggle.com/code/thaimeuu/it5429e-workspace), which uses three Kaggle utility scripts that are based on this repository.

| URL | Description | Based on |
|---|---|---|
https://www.kaggle.com/code/thaimeuu/it5429e-workspace | Experiments (outputs, which are used in the reported, are saved in this notebook) | Independent |
| https://www.kaggle.com/code/thaimeuu/graph-ml-it5429e-models | All models | [models/](./models/) |
| https://www.kaggle.com/code/thaimeuu/graph-ml-it5429e-learning | Training and evaluation functions | Independent |
| https://www.kaggle.com/code/thaimeuu/graph-ml-it5429e-utils | Utils functions | [utils.py](./utils.py) |
