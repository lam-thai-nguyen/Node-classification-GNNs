import os
import sys
import random

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RESULTS_DIR = os.path.join(_THIS_DIR, "results")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


_DATASET_CACHE = {}


def get_dataset(gnn=True):
    from utils import load_dataset

    if gnn not in _DATASET_CACHE:
        _DATASET_CACHE[gnn] = load_dataset(gnn)
    return _DATASET_CACHE[gnn]
