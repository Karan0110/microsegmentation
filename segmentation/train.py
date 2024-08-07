import os
from pathlib import Path
import argparse
import json5
import time
from typing import Iterable, Tuple, Any, Union
import itertools
import shutil
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from epoch import train_model, test_model
from utils.synthetic_dataset import get_data_loaders, Labels
from criterions.utils import get_criterions
from demo import plot_demos
from utils import instantiate_from_dict, load_json5, get_device
from models.serialization import get_model_save_file_paths, save_model_to_file
import models

def get_model(model_config : dict,
              device : torch.device,
              verbose : bool) -> nn.Module:
    model = instantiate_from_dict(models, information=model_config)
    model = model.to(device)

    if verbose:
        print(f"\nCreated {model_name} model.")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"It has {total_params} parameters.")

    return model

def get_optimizer(model : nn.Module,
                  optimizer_config : dict,
                  verbose : bool) -> optim.Optimizer:
    extra_optimizer_config = copy.deepcopy(optimizer_config)
    extra_optimizer_config['params']['params'] = model.parameters()

    optimizer = instantiate_from_dict(namespace=torch.optim, 
                                      information=extra_optimizer_config)

    if verbose:
        print(f"\nInitialized optimizer.")

    return optimizer

def get_scheduler(optimizer : optim.Optimizer,
                  scheduler_config : dict,
                  verbose : bool) -> optim.lr_scheduler.LRScheduler:
    extra_scheduler_config = copy.deepcopy(scheduler_config)
    extra_scheduler_config['params']['optimizer'] = optimizer

    scheduler = instantiate_from_dict(namespace=optim.lr_scheduler,
                                      information=extra_scheduler_config)

    if verbose:
        print("\nInitialized scheduler.")
    
    return scheduler

def get_command_line_args() -> argparse.Namespace:
    # Parse CL arguments
    parser = argparse.ArgumentParser(
        description="Train U-Net model on synthetic training data."
    )

    # Positional argument (mandatory)
    parser.add_argument('-c', '--config', 
                        type=Path, 
                        help='Config Directory (If not in overwrite mode, program uses a pre-existing config if it exists)')

    parser.add_argument('-dc', '--democonfig',
                        type=Path,
                        default=None,
                        help='Path to demo config (If left blank no demos are plotted)')

    parser.add_argument('-n', '--name',
                        type=str,
                        required=True,
                        help="Model name (continues training any pre-existing model)")

    parser.add_argument('-dd', '--datadir',
                        type=Path,
                        required=True,
                        help="Training data directory")

    parser.add_argument('-ld', '--logdir',
                        type=Path,
                        default="runs/",
                        help="Log directory for TensorBoard (Default runs/)")

    parser.add_argument('-md', '--modeldir',
                        type=Path,
                        required=True,
                        help="Model directory")
    
    parser.add_argument('-sm', '--savemode',
                        choices=['best', 'recent'],
                        default='best',
                        help="Which model is saved to file")
    
    parser.add_argument('-e', '--epochs',
                        type=int,
                        help="Number of epochs (Leave blank to continue until KeyboardInterrupt)")

    parser.add_argument('-v', '--verbose', 
                        action='store_true', 
                        help='Increase output verbosity')

    parser.add_argument('--overwrite', 
                        action='store_true', 
                        help='Overwrite saved model, if it exists')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    start_time = time.time()

    # Handle CL arguments
    # --------------------

    args = get_command_line_args()

    verbose = args.verbose
    overwrite_mode = args.overwrite
    save_mode = args.savemode

    model_name = args.name

    model_dir = args.modeldir
    config_path = args.config
    log_dir = args.logdir / model_name
    dataset_dir = args.datadir 
    demo_config_path = args.democonfig

    num_epochs = args.epochs

    # Set up device
    # --------------

    device = get_device(verbose=verbose)

    # Location of model/config savefiles
    # ----------------------------------

    state_save_file_path, \
    config_save_file_path = get_model_save_file_paths(model_dir=model_dir,
                                                      model_name=model_name)

    # Load the config files
    # ---------------------

    config : Union[dict, list]
    if config_save_file_path.exists() and (not overwrite_mode):
        if verbose:
            print(f"\nNot in overwrite mode and a checkpoint config file already exists at {config_save_file_path}.\nUsing checkpoint config dir...")
        config = load_json5(config_save_file_path)
    else:
        print(f"\nConfig file path: {config_path.absolute()}")
        config = load_json5(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file! Must be a dict")

    demo_config = None
    if demo_config_path is not None:
        demo_config = load_json5(demo_config_path)
    if not isinstance(demo_config, dict) and demo_config is not None:
        raise ValueError(f"Invalid demo config file! Must be a dict")

    # Set up TensorBoard writer
    # -------------------------

    if verbose:
        print(f"TensorBoard log dir: {log_dir}")
    writer = SummaryWriter(log_dir)

    # Set up data loaders
    # -------------------
    
    if verbose:
        print(f"Training Data: {dataset_dir}")

    data_config = config['data']
    patch_size = config['model']['patch_size']

    train_loader, test_loader = get_data_loaders(patch_size=patch_size,
                                                 base_dir=dataset_dir,
                                                 augmentation_config=config['augmentation'],
                                                 **data_config,
                                                 verbose=verbose)

    # Initialize model, loss function, optimizer, and scheduler
    # ---------------------------------------------------------

    model = get_model(model_config=config['model'],
                      device=device,
                      verbose=verbose)
    
    criterions = get_criterions(criterions_config=config['criterions'],
                               device=device,
                               Labels=Labels,
                               verbose=verbose)

    optimizer = get_optimizer(model=model,
                              optimizer_config=config['optimizer'],
                              verbose=verbose)

    scheduler = get_scheduler(optimizer=optimizer,
                              scheduler_config=config['scheduler'],
                              verbose=verbose)

    # Check if savefile exists
    # ------------------------
                            
    start_epoch : int
    if state_save_file_path.exists():
        if verbose:
            print(f"\nModel {model_name} savefile found.")
        if not overwrite_mode:
            if verbose:
                print(f"Continuing training of this model from its config file...")
            
            checkpoint = torch.load(state_save_file_path, 
                                    weights_only=True) #safety feature

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            with open(config_save_file_path, 'r') as config_save_file:
                loaded_config = json5.load(config_save_file)
                if not isinstance(loaded_config, dict):
                    raise ValueError(f"Model config path ({config_save_file_path}) is invalid - it should be a dict!")
                config = loaded_config
                start_epoch = config['epoch'] + 1 #type: ignore
        else:
            # Delete tensorboard logs
            if verbose:
                print(f"Clearing old tensorboard logs...")
            shutil.rmtree(writer.log_dir)
            os.makedirs(writer.log_dir, exist_ok=True)

            # Delete pre-existing model files
            if verbose:
                print(f"Clearing old savefiles...")
            os.remove(state_save_file_path)
            os.remove(config_save_file_path)

            start_epoch = 1
    else:
        # Delete any old tensorboard logs
        print(f"Clearing any old tensorboard logs...")
        shutil.rmtree(writer.log_dir)
        os.makedirs(writer.log_dir, exist_ok=True)

        start_epoch = 1

    # Training and testing loop
    # -------------------------

    best_test_loss = None

    epoch_iterator : Iterable
    if num_epochs is not None:
        epoch_iterator = range(start_epoch, start_epoch+num_epochs)
    else:
        epoch_iterator = itertools.count(start=start_epoch)
    for epoch in epoch_iterator:
        try:
            epoch_start_time = time.time()

            print(f"\nEpoch {epoch}")
            print('-'*15 + '\n')

            train_model(model=model,
                        device=device,
                        writer=writer,
                        train_loader=train_loader,
                        criterions=criterions,
                        optimizer=optimizer,
                        epoch=epoch,
                        verbose=verbose)

            test_loss = test_model(model=model, 
                                    device=device, 
                                    writer=writer,
                                    epoch=epoch,
                                    test_loader=test_loader, 
                                    criterions=criterions,
                                    verbose=verbose)

            if verbose:
                print(f"Test Loss: {test_loss}")

            historical_best = False

            if best_test_loss is None or test_loss < best_test_loss:
                if verbose:
                    print(f"This is a historical best test loss.")
                    if save_mode == 'best':
                        print(f"We are in 'best' savemode - so it will be saved to file.")
                best_test_loss = test_loss
                historical_best = True
                    
            scheduler.step()

            if historical_best or save_mode == 'recent':
                save_model_to_file(model=model,
                                config=config,
                                optimizer=optimizer,
                                scheduler=scheduler, 
                                model_dir=model_dir,
                                model_name=model_name,
                                epoch=epoch,
                                verbose=verbose)
                print("\nSaved to file.")

            if demo_config is not None:
                print("\nPlotting demo...")
                plot_demos(demo_config=demo_config,
                           demo_name=f"{model_name}_epoch_{epoch}",
                           model_name=model_name,
                           hard_segmentation_threshold=0.5,
                           model=model,
                           model_dir=model_dir,
                           model_config=config['model'],
                           device=device,
                           verbose=False,
                           only_show_histogram=True,
                           save_to_file=True)

            print(f"\nTook {(time.time() - epoch_start_time) / 60.:.2f} minutes")
        except KeyboardInterrupt:
            print(f"\nInterrupted during epoch {epoch}.")
            break

    # Output time taken
    # -----------------

    time_taken = int(time.time() - start_time)
    seconds = time_taken % 60
    minutes = (time_taken // 60) % 60
    hours = time_taken // (60**2)

    print(f"\nTook {hours} hrs {minutes} min in total.")

    writer.close()
