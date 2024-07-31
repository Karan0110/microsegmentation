# Command line arguments
# dataset_dir       : Path to dataset folder
# save_dir          : Path to model folder
# save_file_name    : Name of save file (without .pth)
# patch_size (=256) : A unique ID for the data sample

import os
from pathlib import Path
import argparse
import json5
import time
from typing import Iterable, Tuple, Any
import itertools
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer

from torch.utils.tensorboard import SummaryWriter

from unet import UNet
from epoch import train_model, test_model
from synthetic_dataset import get_data_loaders
from synthetic_dataset import Labels
from load import load_json5_config
from device import get_device

def get_save_file_paths(model_dir : Path,
                        model_name : str) -> Tuple[Path, Path]:
    state_save_file_path = model_dir / model_name / f"{model_name}.pth"
    config_save_file_path = model_dir / model_name / f"config.json5"

    return state_save_file_path, config_save_file_path

def save_model(model : nn.Module,
               optimizer : Optimizer,
               scheduler : Any,
               model_config : dict,
               training_config : dict,
               augmentation_config : dict,
               training_data_config : dict,
               model_dir : Path, 
               model_name : str,
               epoch : int,
               verbose : bool = False) -> Path:
    model_state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }

    state_save_file_path, config_save_file_path = get_save_file_paths(model_dir=model_dir,
                                                                      model_name=model_name)

    os.makedirs(state_save_file_path.parent, exist_ok=True)
    os.makedirs(config_save_file_path.parent, exist_ok=True)

    try:
        # Save the model state
        torch.save(model_state, state_save_file_path)
        
        # Save the configuration
        with config_save_file_path.open('w') as f:
            json5.dump({
                'epoch': epoch,
                'model': model_config,
                'training': training_config,
                'training_data': training_data_config,
                'augmentation': augmentation_config,
            }, f, indent=4)
    except KeyboardInterrupt:
        # Handle the keyboard interrupt or any cleanup here if necessary
        if verbose:
            print("Training interrupted midway through saving results.")
        # You may optionally delete the partially saved files here if appropriate
        if state_save_file_path.exists():
            state_save_file_path.unlink()
        if config_save_file_path.exists():
            config_save_file_path.unlink()
        raise

    if verbose:
        print(f"Saved model to {state_save_file_path}.")
        print(f"Saved model config to {config_save_file_path}.")
    
    return state_save_file_path

def get_model_criterion_optimizer_scheduler(model_config : dict,
                                            training_config : dict,
                                            device : torch.device,
                                            verbose : bool) -> Tuple[nn.Module,  nn.Module, optim.SGD, Any]:
    depth = model_config['depth']
    in_channels = model_config['in_channels']
    out_channels = model_config['out_channels']
    base_channel_num = model_config['base_channel_num']
    convolution_padding_mode = model_config['convolution_padding_mode']

    model = UNet(depth=depth,
                 base_channel_num=base_channel_num,
                 in_channels=in_channels,
                 out_channels=out_channels,
                 padding_mode=convolution_padding_mode).to(device)

    loss_config = training_config['loss']
    loss_name = loss_config['name']

    raw_loss_params = loss_config['params']
    new_loss_params = raw_loss_params.copy()

    if loss_name == 'CrossEntropyLoss':
        raw_class_weights = raw_loss_params['weight']
        class_weights = [None] * len(Labels)
        for label_str in raw_class_weights:
            index = Labels[label_str].value
            class_weights[index] = raw_class_weights[label_str]
        new_loss_params['weight'] = torch.tensor(class_weights).to(device)

    LossFunction = getattr(nn, loss_name)
    criterion = LossFunction(**new_loss_params)

    optimizer_config = training_config['optimizer']
    optimizer = optim.SGD(model.parameters(), **optimizer_config)

    scheduler_config = training_config['scheduler']
    scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_config)

    return model, criterion, optimizer, scheduler

if __name__ == '__main__':
    start_time = time.time()

    # Parse CL arguments
    parser = argparse.ArgumentParser(
        description="Train U-Net model on synthetic training data."
    )

    # Positional argument (mandatory)
    parser.add_argument('-c', '--config', 
                        type=Path, 
                        required=True,
                        help='Config Directory')

    parser.add_argument('-n', '--name',
                        type=str,
                        required=True,
                        help="Model name (continues training any pre-existing model)")
    
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

    # python3 train.py -c config/ -n model-v3-test -e 5 -v 

    # Verbose CL arg
    verbose = args.verbose

    # Overwrite mode CL arg
    overwrite_mode = args.overwrite

    # Load the config files
    config_dir = args.config

    training_config = load_json5_config(config_dir / 'training-config.json5')
    training_data_config = load_json5_config(config_dir / 'training-data-config.json5')
    augmentation_config = load_json5_config(config_dir / 'augmentation-config.json5')
    model_config = load_json5_config(config_dir / 'model-config.json5')

    # Prepare directories for reading/writing
    model_name = args.name

    dataset_dir = Path(training_data_config['dataset_dir'])
    model_dir = Path(training_config['model_dir']) 

    # Set up TensorBoard writer

    writer = SummaryWriter(f"runs/{model_name}")

    # Set up device
    device : torch.device = get_device(verbose=verbose)

    # Set up data loaders

    # TODO TESTING REMOVE THIS!!!
    # transform_config = augmentation_config['transforms']
    transform_config = []
    # REMOVE THIS!!!
    
    color_to_label = training_data_config['color_to_label']
    color_to_label = {int(key) : value for key, value in color_to_label.items()}

    batch_size = training_config['batch_size']
    train_test_split = training_config['train_test_split']
    batches_per_epoch = training_config['batches_per_epoch']
    batches_per_test = training_config['batches_per_test']

    patch_size = model_config['patch_size']

    train_loader, test_loader = get_data_loaders(patch_size=patch_size,
                                                 base_dir=dataset_dir,
                                                 transform_config=transform_config,
                                                 color_to_label=color_to_label,
                                                 batch_size=batch_size,
                                                 train_test_split=train_test_split)

    # Initialize model, loss function, optimizer, and scheduler

    model, criterion, optimizer, scheduler = get_model_criterion_optimizer_scheduler(model_config=model_config,
                                                                                     training_config=training_config,
                                                                                     device=device,
                                                                                     verbose=verbose)
    start_epoch : int                                                                                

    # Check if model is pre-existing, if so continue working on it

    state_save_file_path, config_save_file_path = get_save_file_paths(model_dir=model_dir,
                                                                      model_name=model_name)
                            
    if state_save_file_path.exists():
        if verbose:
            print(f"Model {model_name} savefile found.")
        if not overwrite_mode:
            if verbose:
                print(f"Continuing training of this model from its config file...")
            
            checkpoint = torch.load(state_save_file_path, 
                                    weights_only=True) #safety feature

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            with open(config_save_file_path, 'r') as config_save_file:
                config = json5.load(config_save_file)
                start_epoch = config['epoch'] + 1 #type: ignore
        else:
            if verbose:
                print(f"Overwriting savefile with new run...")

            # Delete tensorboard logs
            shutil.rmtree(writer.log_dir)
            os.makedirs(writer.log_dir, exist_ok=True)

            # Delete pre-existing model files
            os.remove(state_save_file_path)
            os.remove(config_save_file_path)

            start_epoch = 1
    else:
        # Delete any old tensorboard logs
        shutil.rmtree(writer.log_dir)
        os.makedirs(writer.log_dir, exist_ok=True)

        start_epoch = 1

    # Training and testing loop

    num_epochs = args.epochs

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
                        criterion=criterion,
                        optimizer=optimizer,
                        epoch=epoch,
                        num_batches=batches_per_epoch,
                        verbose=verbose)

            test_model(model=model, 
                       device=device, 
                       writer=writer,
                       epoch=epoch,
                       patch_size=patch_size,
                       test_loader=test_loader, 
                       criterion=criterion,
                       num_batches=batches_per_test,
                       verbose=verbose)
                    
            scheduler.step()

            save_model(model=model,
                    optimizer=optimizer,
                    scheduler=scheduler, 
                    model_config=model_config,
                    training_config=training_config,
                    augmentation_config=augmentation_config,
                    training_data_config=training_data_config,
                    model_dir = model_dir,
                    model_name=model_name,
                    epoch=epoch,
                    verbose=verbose)

            print(f"Took {(time.time() - epoch_start_time) / 60.:.2f} minutes")
        except KeyboardInterrupt:
            print(f"Interrupted during epoch {epoch}.")
            break

    # Output time taken
    time_taken = int(time.time() - start_time)
    seconds = time_taken % 60
    minutes = (time_taken // 60) % 60
    hours = time_taken // (60**2)

    print(f"\nTook {hours} hrs {minutes} min in total.")

    writer.close()

# python3 train.py train-config.json5 augmentation-config.json5 model-v3
