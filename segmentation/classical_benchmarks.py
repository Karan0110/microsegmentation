from pathlib import Path
import os
import random
from typing import Tuple, Union, List, Optional
from dotenv import load_dotenv
import argparse

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn

from utils import get_device

from data.synthetic_dataset import get_data_loaders
from criterions.utils import get_criterions

from global_utils.arguments import get_path_argument
from global_utils import load_json5

from pprint import pprint

def parse_args() -> argparse.Namespace:
    # Parse CL arguments
    parser = argparse.ArgumentParser(
        description="Calculate segmentation benchmarks using classical non-ML techniques"
    )

    # Positional argument (mandatory)
    parser.add_argument('-i', '--inputdir', 
                        type=Path, 
                        help='Benchmark image dir (Leave blank to use environment variable value)')

    parser.add_argument('-c', '--config', 
                        type=Path, 
                        help='Config Directory (Leave blank to use environment variable value)')

    parser.add_argument('-v', '--verbose', 
                        action='store_true', 
                        help='Increase output verbosity')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    load_dotenv()

    verbose = args.verbose

    device = get_device(verbose=verbose)

    config_path = get_path_argument(cl_args=args,
                                    cl_arg_name='config',
                                    env_var_name='CONFIG_PATH')
    config = load_json5(config_path)

    dataset_dir = get_path_argument(cl_args=args,
                                    cl_arg_name='inputdir',
                                    env_var_name='DATA_PATH')
    if verbose:
        print(f"Dataset path: {dataset_dir}")

    data_config = config['data']

    train_loader, test_loader = get_data_loaders(patch_size=config['model']['patch_size'],
                                                 base_dir=dataset_dir,
                                                 augmentation_config=config['augmentation'],
                                                 **data_config,
                                                 num_workers=0,
                                                 verbose=verbose)
    
    inputs, targets = next(iter(train_loader))

    plt.imshow(inputs.squeeze().to('cpu').numpy(), cmap='gray')
    plt.show()

    plt.imshow(targets.squeeze().to('cpu').numpy(), cmap='gray')
    plt.show()

    criterions = get_criterions(criterions_config=config['criterions'],
                               device=device,
                               verbose=verbose)

    # Feeding in input as segmentation
    logits = torch.log(inputs / (1.-inputs) + 1e-6)
    outputs = torch.cat((logits, torch.zeros_like(logits)), dim=1)

    loss = {}
    for criterion in criterions:
        # print("TESTING TODO")
        # print(criterion['criterion'])
        # print(outputs.shape)
        # print(targets.shape)
        # print(torch.unique(targets))

        loss[criterion['name']] = criterion['criterion'](outputs, targets)

    print()
    pprint(loss)



