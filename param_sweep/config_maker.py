from load_json5 import load_json5
import os
import json5
from pathlib import Path
import itertools
import copy

base_dir = Path("/Users/karan/microsegmentation")
original = load_json5(base_dir / "segmentation/config")

depths = [4, 5]
dropout_rates = [0.2, 0.5]
lrs = [1e-6, 1e-5, 1e-4]
tversky_betas = [0.5, 0.7]

for beta, depth, dropout_rate, lr in itertools.product(tversky_betas, depths, dropout_rates, lrs):
    new_config = copy.deepcopy(original)

    new_config['model']['params']['depth'] = depth #type: ignore
    new_config['model']['params']['dropout_rate'] = dropout_rate #type: ignore
    new_config['optimizer']['params']['lr'] = lr #type: ignore
    new_config['criterions'][1]['params']['beta'] = beta #type: ignore

    dropout_name = None
    if dropout_rate == 0.2:
        dropout_name = 'low'
    elif dropout_rate == 0.5:
        dropout_name = 'high'

    if lr == 1e-6:
        lr_name = 'low'
    elif lr == 1e-5:
        lr_name = 'medium'
    elif lr == 1e-4:
        lr_name = 'high'

    path = Path("./config_files_test") / f"depth_{depth}__dropout_{dropout_name}__lr_{lr_name}__tbeta_{beta}.json5" #type: ignore
    os.makedirs(path.parent, exist_ok=True)

    with open(path, 'w') as file:
        json5.dump(new_config, file, indent=4)