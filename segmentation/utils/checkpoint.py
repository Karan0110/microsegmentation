from pathlib import Path
from typing import Tuple, Any, Optional
import os

import torch
import torch.nn as nn

from global_utils import load_json5, save_json5
from criterions.utils import get_criterions

from .serialization import load_model, load_optimizer, load_scheduler, save_model, save_optimizer, save_scheduler

def save_checkpoint(save_dir : Path,
                    config : dict,
                    model : nn.Module,
                    optimizer : torch.optim.Optimizer,
                    scheduler : torch.optim.lr_scheduler.LRScheduler,
                    verbose : bool = False) -> None:
    config_file_path = save_dir / 'config.json5'
    save_json5(data=config,
               path=config_file_path,
               pretty_print=True)
    if verbose:
        print(f"Saved config file to {config_file_path}")

    save_model(model=model,
               save_dir=save_dir,
               verbose=verbose)

    save_optimizer(optimizer=optimizer,
                   save_dir=save_dir,
                   verbose=verbose)

    save_scheduler(scheduler=scheduler,
                   save_dir=save_dir,
                   verbose=verbose)


def load_checkpoint(device : torch.device,
                    config_path : Optional[Path] = None,
                    save_dir : Optional[Path] = None,
                    verbose : bool = False) -> dict:
    if save_dir is not None and save_dir.exists():
        config_path = save_dir / 'config.json5'
        if verbose:
            print(f"Checkpoint found at {save_dir}. Using config: {config_path}")
        is_new = False
    else:
        if verbose:
            print(f"No checkpoint found. Using config: {config_path}") 
        if config_path is None:
            raise ValueError(f"Must provide either an existing save_dir (to load from checkpoint) or config_path (to instantiate checkpoint)")
        is_new = True

    config : dict
    config = load_json5(config_path)

    if verbose:
        print()

    model = load_model(model_config=config['model'],
                       device=device,
                       save_dir=save_dir,
                       verbose=verbose)
    
    criterions = get_criterions(criterions_config=config['criterions'],
                               device=device,
                               verbose=verbose)

    optimizer = load_optimizer(optimizer_config=config['optimizer'],
                               model=model,
                               save_dir=save_dir,
                               verbose=verbose)

    scheduler = load_scheduler(scheduler_config=config['scheduler'],
                               optimizer=optimizer,
                               save_dir=save_dir,
                               verbose=verbose)

    return {
        'model': model,
        'criterions': criterions,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'config': config,
        'is_new': is_new,
    }