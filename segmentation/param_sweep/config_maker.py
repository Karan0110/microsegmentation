from global_utils import load_json5
import os
import json5
from pathlib import Path
import itertools
import copy

base_dir = Path("/Users/karan/microsegmentation")
original = load_json5(base_dir / "segmentation/config")

# depths = [4, 5]
# dropout_rates = [0.2, 0.5]
# lrs = [1e-6, 1e-5, 1e-4]

# tversky_betas = [0.5, 0.7]

loss_weights = [0.5, 0.6, 0.7]

names = ['Heather', 'Veronica', 'Martha']

for i, loss_weight in enumerate(loss_weights):
    name = names[i]

    new_config = copy.deepcopy(original)

    if not isinstance(new_config, dict):
        raise ValueError(f"Config file should be a dict!")

    new_config['criterions'][1]['weight'] = loss_weight 
    new_config['criterions'][0]['weight'] = 1. - loss_weight

    path = Path("./configs") / f"{name}.json5"
    os.makedirs(path.parent, exist_ok=True)
    with open(path, 'w') as file:
        json5.dump(new_config, file, indent=4)

    # Making test configs
    new_config['data']['max_batches_per_train_epoch'] = 2
    new_config['data']['max_batches_per_test'] = 2

    new_config['model']['depth'] = 2
    new_config['model']['base_channel_num'] = 1

    path = Path("./test_configs") / f"{name}_TEST.json5"
    os.makedirs(path.parent, exist_ok=True)
    with open(path, 'w') as file:
        json5.dump(new_config, file, indent=4)
