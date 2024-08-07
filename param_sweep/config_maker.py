from load_json5 import load_json5
import json5
from pathlib import Path
import itertools
import copy

original = load_json5(Path("/Users/karan/microsegmentation/segmentation/config"))

depths = [4, 5]
dropout_rates = [0.2, 0.5]
lrs = [1e-6, 1e-5, 1e-4]

for depth, dropout_rate, lr in itertools.product(depths, dropout_rates, lrs):
    new_config = copy.deepcopy(original)

    new_config['model']['params']['depth'] = depth
    new_config['model']['params']['dropout_rate'] = dropout_rate
    new_config['optimizer']['params']['lr'] = lr

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

    path = Path("./config_files") / f"depth_{depth}__dropout_{dropout_name}__lr_{lr_name}.json5"

    with open(path, 'w') as file:
        json5.dump(new_config, file, indent=4)