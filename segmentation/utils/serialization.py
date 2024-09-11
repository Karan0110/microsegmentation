from pathlib import Path
from typing import Tuple, Any, Optional, List
import os
import copy

import torch
import torch.nn as nn

from global_utils import load_json5, save_json5, instantiate_from_dict
from criterions.utils import get_criterions

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

def load_model(device : torch.device,
               verbose : bool,
               model_dir : Path,
               config : Optional[dict] = None) -> nn.Module:

    model = _init_or_load_checkpoint(device=device,
                                          verbose=verbose,
                                          model_dir=model_dir,
                                          config=config,
                                          model_only=True)['model']

    return model

def load_checkpoint(device : torch.device,
                    verbose : bool,
                    model_dir : Path,
                    config : Optional[dict] = None) -> dict:

    checkpoint = _init_or_load_checkpoint(device=device,
                                          verbose=verbose,
                                          model_dir=model_dir,
                                          config=config,
                                          model_only=False)

    return checkpoint

def init_checkpoint(device : torch.device,
                    verbose : bool,
                    config : dict) -> dict:

    checkpoint = _init_or_load_checkpoint(device=device,
                                          verbose=verbose,
                                          config=config,
                                          model_only=False)

    return checkpoint

def _init_or_load_checkpoint(device : torch.device,
                             verbose : bool,
                             model_only : bool,
                             model_dir : Optional[Path] = None,
                             config : Optional[dict] = None) -> dict:
    if model_dir is None and config is None:
        raise ValueError("Must either provide a model dir path (load) or a config (init)")

    if verbose:
        print()

    # Declare elements of checkpoint
    model : nn.Module
    optimizer : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler.LRScheduler
    criterions : List[dict]

    # Load config file
    if config is None:
        if model_dir is None:
            raise ValueError("Must either provide a model dir path (load) or a config file path (init)")
        config_file_path = model_dir / 'config.json5'
        config = load_json5(config_file_path)

        if verbose:
            print(f"Loading config file: {config_file_path}")

    if not isinstance(config, dict):
        raise ValueError("Config must be a dict!")

    # Init model, optimizer, scheduler, criterions
    if verbose:
        if model_only:
            print(f"Initializing model")
        else:
            print(f"Initializing model, optimizer, scheduler and criterions...")

    model = instantiate_from_dict(models,
                                  information=config['model'])

    if not model_only:
        criterions = get_criterions(config=config,
                                    device=device,
                                    verbose=verbose)

        optimizer_config = copy.deepcopy(config['optimizer'])
        optimizer_config['params']['params'] = model.parameters()
        optimizer = instantiate_from_dict(torch.optim,
                                        information=optimizer_config)

        scheduler_config = copy.deepcopy(config['scheduler'])
        scheduler_config['params']['optimizer'] = optimizer
        scheduler = instantiate_from_dict(torch.optim.lr_scheduler,
                                        information=scheduler_config)

    # Load states from checkpoint files

    if model_dir is not None:
        if verbose:
            print(f"Loading states of model, optimizer and scheduler from checkpoint...")

        model_state_dict : dict = torch.load(model_dir / 'model.pth',
                                            weights_only=True,
                                            map_location=device)
        model.load_state_dict(model_state_dict)

        if not model_only:
            optimizer_state_dict : dict = torch.load(model_dir / 'optimizer.pth',
                                                    weights_only=True)
            optimizer.load_state_dict(optimizer_state_dict) #type: ignore

            scheduler_state_dict : dict = torch.load(model_dir / 'scheduler.pth',
                                                    weights_only=True)
            scheduler.load_state_dict(scheduler_state_dict) #type: ignore

    # Move to specified device

    model = model.to(device)

    if model_only:
        return {
            'model': model,
        }
    else:
        return {
            'model': model,
            'criterions': criterions, #type: ignore
            'optimizer': optimizer, #type: ignore
            'scheduler': scheduler, #type: ignore
        }
