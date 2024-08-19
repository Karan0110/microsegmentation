import os
from pathlib import Path
import argparse
import json5
import time
from typing import Iterable, Tuple, Any, Union, List, Optional
import itertools
import shutil
import copy
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from epoch import train_model, test_model
from data.synthetic_dataset import get_data_loaders, Labels
from global_utils import load_json5, save_json5
from global_utils.arguments import get_argument, get_path_argument
from utils import get_device

from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.serialization import save_model

def get_command_line_args() -> argparse.Namespace:
    # Parse CL arguments
    parser = argparse.ArgumentParser(
        description="Train U-Net model on synthetic training data."
    )

    # Positional argument (mandatory)
    parser.add_argument('-e', '--epochs',
                        type=int,
                        help="Number of epochs (Leave blank to continue until KeyboardInterrupt)")

    parser.add_argument('-v', '--verbose', 
                        action='store_true', 
                        help='Increase output verbosity')

    parser.add_argument('-n', '--name',
                        type=str,
                        required=True,
                        help="Model name (continues training any pre-existing model if checkpoint exists)")

    parser.add_argument('-nw', '--numworkers',
                        type=int,
                        help="Number of workers for data loader (Leave blank if you don't know what this means)")

    parser.add_argument('-w', '--weights',
                        type=str,
                        help="Begin training from another model's weights (This option cannot be used when training from a checkpoint)")

    parser.add_argument('-c', '--config', 
                        type=Path, 
                        help='Config Directory (Ignores and uses a pre-existing config if it checkpoint exists)')

    parser.add_argument('-dd', '--datadir',
                        type=Path,
                        help="Training data directory")

    parser.add_argument('-ld', '--logdir',
                        type=Path,
                        help="Log directory for TensorBoard")

    parser.add_argument('-md', '--modeldir',
                        type=Path,
                        help="Model directory")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    start_time = time.time()

    # Handle CL arguments
    # --------------------

    load_dotenv()
    args = get_command_line_args()

    verbose : bool
    verbose = args.verbose

    model_name : str
    model_name = args.name

    num_epochs : int
    num_epochs = args.epochs

    model_dir = get_path_argument(cl_args=args,
                                  cl_arg_name='modeldir',
                                  env_var_name='MODELS_PATH')
    model_dir = model_dir / model_name

    num_workers : Optional[int]
    num_workers = get_argument(cl_args=args,
                               cl_arg_name='numworkers',
                               env_var_name='NUM_WORKERS',
                               ArgumentType=int)
    
    initial_weight_model_name : str
    initial_weight_model_name = args.weights

    config_path = get_path_argument(cl_args=args,
                                    cl_arg_name='config',
                                    env_var_name='CONFIG_PATH')

    log_dir = get_path_argument(cl_args=args,
                                cl_arg_name='logdir',
                                env_var_name='LOG_PATH')
    log_dir = log_dir / model_name

    dataset_dir = get_path_argument(cl_args=args,
                                    cl_arg_name='datadir',
                                    env_var_name='DATA_PATH')

    # Set up device
    # --------------

    device = get_device(verbose=verbose)

    # Set up TensorBoard writer
    # -------------------------

    if verbose:
        print(f"\nTensorBoard log dir: {log_dir}")
    writer = SummaryWriter(log_dir)

    # Initialize model, loss function, optimizer, and scheduler
    # ---------------------------------------------------------

    checkpoint = load_checkpoint(device=device,
                                 config_path=config_path,
                                 save_dir=model_dir,
                                 verbose=verbose)
    model : nn.Module
    optimizer : torch.optim.Optimizer
    criterions : List[dict]
    scheduler : torch.optim.lr_scheduler.LRScheduler
    config : dict

    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    criterions = checkpoint['criterions']
    scheduler = checkpoint['scheduler']
    config = checkpoint['config']

    # Use pre-existing weights
    # ------------------------

    is_new = checkpoint['is_new']
    if initial_weight_model_name is not None:
        if not is_new:
            raise ValueError(f"An initial model ({initial_weight_model_name}) was provided to use as initial weights.\nThis is not allowed when training from a checkpoint!")

        initial_state_file_path = model_dir.parent / initial_weight_model_name / 'model.pth'

        initial_state_dict : dict = torch.load(initial_state_file_path,
                                               weights_only=True,
                                               map_location=device)
        model.load_state_dict(initial_state_dict)

        if verbose:
            print(f"\nUsing initial weights from model: {initial_weight_model_name}")

    # Set up data loaders
    # -------------------
    
    if verbose:
        print(f"\nTraining Data: {dataset_dir}")

    data_config = config['data']
    patch_size = config['model']['patch_size']
    if verbose:
        print(f"\nPatch size: {patch_size}")

    train_loader, test_loader = get_data_loaders(patch_size=patch_size,
                                                 base_dir=dataset_dir,
                                                 augmentation_config=config['augmentation'],
                                                 **data_config,
                                                 num_workers=num_workers,
                                                 verbose=verbose)

    # Training and testing loop
    # -------------------------

    is_new = checkpoint['is_new']
    if is_new:
        print(f"\nClearing any old tensorboard logs...")
        shutil.rmtree(writer.log_dir)
        os.makedirs(writer.log_dir, exist_ok=True)

    start_epoch = config.get('epoch', 0) + 1

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

            # Check if it's the best model so far (and if so save to file)
            if best_test_loss is None or test_loss < best_test_loss:
                if verbose:
                    print(f"This is a historical best test loss.")

                best_test_loss = test_loss

                if verbose:
                    print()
                save_model(model=model,
                           save_dir=model_dir / 'best',
                           verbose=verbose)
                save_json5(data={**config,
                                 'epoch': epoch,},
                            path=model_dir / 'best' / 'config.json5',
                            pretty_print=True)
                
            scheduler.step()

            save_checkpoint(save_dir=model_dir,
                            config={**config,
                                    'epoch': epoch,},
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            verbose=verbose)

            if verbose:
                print("\nSaved checkpoint to file.")
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
