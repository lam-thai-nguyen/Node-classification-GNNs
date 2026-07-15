import common  # noqa: F401
from models import (
    GCN,
    GCNSkip,
    GCNLayer,
    GCNSL,
    SAGE,
    SAGESkip,
    SAGELayer,
    SAGESL,
)

OUT_CHANNELS = 40

GCN_VARIANTS = {
    "GCN": GCN,
    "GCN-Skip": GCNSkip,
    "GCN-Layer": GCNLayer,
    "GCN-SL": GCNSL,
}

SAGE_VARIANTS = {
    "SAGE": SAGE,
    "SAGE-Skip": SAGESkip,
    "SAGE-Layer": SAGELayer,
    "SAGE-SL": SAGESL,
}

ALL_VARIANTS = {**GCN_VARIANTS, **SAGE_VARIANTS}

MAIN_RESULT = {
    "hidden_channels": 128,
    "dropout": 0.5,
    "lr": 0.01,
    "epochs": 300,
    "n_runs": 3,
    "num_layers": {
        "GCN": 3,
        "GCN-Skip": 5,
        "GCN-Layer": 5,
        "GCN-SL": 5,
        "SAGE": 3,
        "SAGE-Skip": 5,
        "SAGE-Layer": 5,
        "SAGE-SL": 5,
    },
}

OVERSMOOTHING = {
    "hidden_channels": 64,
    "dropout": 0.5,
    "lr": 0.01,
    "epochs": 100,
    "depths": [2, 10, 20, 50],
}

ABLATION = {
    "hidden_channels": 128,
    "dropout": 0.5,
    "lr": 0.01,
    "epochs": 300,
    "n_runs": 3,
    "num_layers": 3,
}
