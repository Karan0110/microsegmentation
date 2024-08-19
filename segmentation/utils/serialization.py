from pathlib import Path
import copy
from typing import Tuple, Any, Optional
import os

import torch
import torch.nn as nn

from global_utils import instantiate_from_dict
import segmentation.models as models

def save_scheduler(scheduler : torch.optim.lr_scheduler.LRScheduler,
                   save_dir : Path,
                   scheduler_file_name : str = 'scheduler',
                   verbose : bool = False) -> None:
    scheduler_state = scheduler.state_dict()
    scheduler_file_path = save_dir / f"{scheduler_file_name}.pth"

    os.makedirs(save_dir, exist_ok=True)

    # Save the scheduler state
    torch.save(scheduler_state, scheduler_file_path)

    if verbose:
        print(f"Saved scheduler state to {scheduler_file_path}.")

def load_scheduler(scheduler_config : dict,
                   optimizer : torch.optim.Optimizer,
                   scheduler_file_name : str = 'scheduler',
                   save_dir : Optional[Path] = None,
                   verbose : bool = False) -> torch.optim.lr_scheduler.LRScheduler:
    extra_scheduler_config = copy.deepcopy(scheduler_config)
    extra_scheduler_config['params']['optimizer'] = optimizer

    scheduler : torch.optim.lr_scheduler.LRScheduler
    scheduler = instantiate_from_dict(torch.optim.lr_scheduler,
                                      information=extra_scheduler_config)

    if save_dir is not None and (scheduler_file_path := save_dir / f"{scheduler_file_name}.pth").exists():
        state_dict : dict = torch.load(scheduler_file_path,
                                       weights_only=True)
        scheduler.load_state_dict(state_dict)
        if verbose:
            print(f"Loaded scheduler state from checkpoint file: {scheduler_file_path}")
    else:
        if verbose:
            print(f"No scheduler checkpoint found. Using fresh initial state")

    return scheduler

def save_optimizer(optimizer : torch.optim.Optimizer,
                   save_dir : Path,
                   optimizer_file_name : str = 'optimizer',
                   verbose : bool = False) -> None:
    optimizer_state = optimizer.state_dict()
    optimizer_file_path = save_dir / f"{optimizer_file_name}.pth"

    os.makedirs(save_dir, exist_ok=True)

    # Save the optimizer state
    torch.save(optimizer_state, optimizer_file_path)

    if verbose:
        print(f"Saved optimizer state to {optimizer_file_path}.")

def load_optimizer(optimizer_config : dict,
                   model : nn.Module,
                   save_dir : Optional[Path] = None,
                   optimizer_file_name : str = 'optimizer',
                   verbose : bool = False) -> torch.optim.Optimizer:
    extra_optimizer_config = copy.deepcopy(optimizer_config)
    extra_optimizer_config['params']['params'] = model.parameters()

    optimizer : torch.optim.Optimizer
    optimizer = instantiate_from_dict(torch.optim,
                                  information=extra_optimizer_config)

    if save_dir is not None and (optimizer_file_path := save_dir / f"{optimizer_file_name}.pth").exists():
        state_dict : dict = torch.load(optimizer_file_path,
                                       weights_only=True)
        optimizer.load_state_dict(state_dict)
        if verbose:
            print(f"Loaded optimizer state from checkpoint file: {optimizer_file_path}")
    else:
        if verbose:
            print(f"No optimizer checkpoint found. Using fresh initial state")

    return optimizer

def save_model(model : nn.Module,
                save_dir : Path, 
                model_file_name : str = 'model',
                verbose : bool = False) -> None:
    model_state = model.state_dict()
    model_file_path = save_dir / f"{model_file_name}.pth"

    os.makedirs(save_dir, exist_ok=True)

    # Save the model state
    torch.save(model_state, model_file_path)

    if verbose:
        print(f"Saved model state to {model_file_path}.")

def load_model(model_config : dict,
               device : torch.device,
               model_file_name : str = 'model',
               save_dir : Optional[Path] = None,
               verbose : bool = False) -> nn.Module:
    model : nn.Module
    model = instantiate_from_dict(models,
                                  information=model_config)

    if save_dir is not None and (model_file_path := save_dir / f"{model_file_name}.pth").exists():
        state_dict : dict = torch.load(model_file_path,
                                    weights_only=True,
                                    map_location=device)
        model.load_state_dict(state_dict)
        if verbose:
            print(f"Loaded model weights from {model_file_name}")
    else:
        if verbose:
            print(f"Initialized model with random weights.")

    model = model.to(device)

    return model


