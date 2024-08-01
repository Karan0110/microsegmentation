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

from torch.utils.tensorboard import SummaryWriter

from unet import UNet
from epoch import train_model, test_model
from synthetic_dataset import get_data_loaders
from loss import get_criterion

from device import get_device
from file_io import load_json5_config, get_model_save_file_paths, save_model

def create_model_criterion_optimizer_scheduler(model_config : dict,
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
    criterion = get_criterion(loss_config=loss_config,
                              device=device)

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
                        help='Config Directory (If not in overwrite mode, program uses a pre-existing config if it exists)')

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

    # Load config files from local directory or model if we are using checkpoint
    model_name = args.name
    model_dir = args.modeldir

    state_save_file_path, config_save_file_path = get_model_save_file_paths(model_dir=model_dir,
                                                                            model_name=model_name)

    # Load the config files
    config_dir = args.config

    if config_save_file_path.exists() and (not overwrite_mode):
        if verbose:
            print(f"\nNot in overwrite mode and a checkpoint config file already exists at {config_save_file_path}.\nUsing checkpoint config dir...")

        config = load_json5_config(config_save_file_path)

        training_config = config['training']
        training_data_config = config['training_data']
        augmentation_config = config['augmentation']
        model_config = config['model']
    else:
        print(f"\nConfig files: {config_dir.absolute()}")

        training_config = load_json5_config(config_dir / 'training-config.json5')
        training_data_config = load_json5_config(config_dir / 'training-data-config.json5')
        augmentation_config = load_json5_config(config_dir / 'augmentation-config.json5')
        model_config = load_json5_config(config_dir / 'model-config.json5')

    # Set up TensorBoard writer
    log_dir = args.logdir / f"{model_name}"
    if verbose:
        print(f"TensorBoard log dir: {log_dir}")
    writer = SummaryWriter(log_dir)
    
    dataset_dir = args.datadir 
    print(f"Training Data: {dataset_dir}")

    # Set up device
    device : torch.device = get_device(verbose=verbose)

    #CHECKPOINT - working till here!

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

    model, criterion, optimizer, scheduler = create_model_criterion_optimizer_scheduler(model_config=model_config,
                                                                                        training_config=training_config,
                                                                                        device=device,
                                                                                        verbose=verbose)

    # Check if savefile exists: if so, continue training from checkpoint
                            
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
# python3 train.py 
# python train.py -c config/ --name model-TEST -dd /Users/karan/MTData/Synthetic_CLEAN -md /Users/karan/microsegmentation/Models --verbose --epoch 5 
