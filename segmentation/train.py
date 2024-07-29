# Command line arguments
# dataset_dir       : Path to dataset folder
# save_dir          : Path to model folder
# save_file_name    : Name of save file (without .pth)
# patch_size (=256) : A unique ID for the data sample

import os
import sys
from pathlib import Path
import json5
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer, lr_scheduler

from torch.utils.tensorboard import SummaryWriter

from unet import UNet
from epoch import train_model, test_model
from synthetic_dataset import get_data_loaders
from synthetic_dataset import Labels
from demo import log_demo

def load_json5_config(file_path : Path) -> dict:
    if file_path.suffix != '.json5':
        raise ValueError(f"Invalid config file path {file_path}\n" + f"Config file must be .json5")

    with file_path.open('r') as file:
        config = json5.load(file)
    if not isinstance(config, dict):
        raise TypeError(f"JSON5 config file {file_path} is of invalid format! It should be a dictionary")

    return config

def save_model(model : nn.Module,
               optimizer : Optimizer,
               scheduler : lr_scheduler._LRScheduler,
               model_config : dict,
               training_config : dict,
               save_file_dir : Path, 
               file_name_stem : str,
               verbose : bool = False) -> Path:
    model_state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }

    os.makedirs(save_file_dir, exist_ok=True)

    state_save_file_path = save_file_dir / f"{file_name_stem}.pth"
    config_save_file_path = save_file_dir / f"{file_name_stem}_config.pkl"

    torch.save(model_state,
                f=state_save_file_path)
    with config_save_file_path.open('wb') as f:
        pickle.dump({
            'model': model_config,
            'training': training_config,
        }, f)

    if verbose:
        print(f"Saved model to {state_save_file_path}.")
        print(f"Saved model config to {config_save_file_path}.")
    
    return state_save_file_path

if __name__ == '__main__':
    start_time = time.time()

    # Validate command line arguments
    if (len(sys.argv)-1) not in [1,2]:
        print("Too few / many command-line arguments!")
        print("Correct usage of program:")
        print(f"python3 {sys.argv[0]} [config_dir] <save_file_name>")
        print("Where default name for model save file is \"model\"")
        exit(1)

    # Load the config files
    config_dir = Path(sys.argv[1])

    training_config = load_json5_config(config_dir / 'training-config.json5')
    training_data_config = load_json5_config(config_dir / 'training-data-config.json5')
    augmentation_config = load_json5_config(config_dir / 'augmentation-config.json5')
    demo_config = load_json5_config(config_dir / 'demo-config.json5')
    model_config = load_json5_config(config_dir / 'model-config.json5')

    # Prepare directories for reading/writing
    save_file_name = sys.argv[2] if (len(sys.argv)-1) == 2 else "model"

    dataset_dir = Path(training_data_config['dataset_dir'])
    save_file_dir = Path(training_config['save_file_dir']) 

    os.makedirs(save_file_dir, exist_ok=True)

    writer = SummaryWriter(f"runs/{save_file_name}")

    # General device handling: Check for CUDA/GPU, else fallback to CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = 'Apple Silicon GPU'
    else:
        device = torch.device('cpu')
        device_name = 'CPU'
    print(f"Using device: {device}")

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

    depth = model_config['depth']
    in_channels = model_config['in_channels']
    out_channels = model_config['out_channels']
    base_channel_num = model_config['base_channel_num']
    convolution_padding_mode = model_config['convolution_padding_mode']

    num_epochs = training_config['num_epochs']

    model = UNet(depth=depth,
                 base_channel_num=base_channel_num,
                 in_channels=in_channels,
                 out_channels=out_channels,
                 padding_mode=convolution_padding_mode).to(device)

    loss_config = training_config['loss']
    loss_name = loss_config['name']
    loss_params = loss_config['params']

    if loss_name == 'CrossEntropyLoss':
        raw_class_weights = loss_params['weight']
        class_weights = [None] * len(Labels)
        for label_str in raw_class_weights:
            index = Labels[label_str].value
            class_weights[index] = raw_class_weights[label_str]
        loss_params['weight'] = torch.tensor(class_weights).to(device)

    LossFunction = getattr(nn, loss_name)
    criterion = LossFunction(**loss_params)

    # Training and testing loop

    optimizer_config = training_config['optimizer']
    optimizer = optim.SGD(model.parameters(), **optimizer_config)

    scheduler_config = training_config['scheduler']
    scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_config)


    for epoch in range(num_epochs):
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
                    verbose=True)

        test_model(model=model, 
                   device=device, 
                   writer=writer,
                   epoch=epoch,
                   patch_size=patch_size,
                   test_loader=test_loader, 
                   criterion=criterion,
                   num_batches=batches_per_test,
                   verbose=True)
                   
        scheduler.step()

        save_model(model=model,
                   optimizer=optimizer,
                   scheduler=scheduler, #type: ignore
                   model_config=model_config,
                   training_config=training_config,
                   save_file_dir=save_file_dir / save_file_name,
                   file_name_stem=f"epoch_{epoch+1}",
                   verbose=True)

        print(f"Took {(time.time() - epoch_start_time) / 60.:.2f} minutes")

    print("\nSaving final model to file...")
    state_save_file_path = save_model(model=model,
                                      optimizer=optimizer,
                                      scheduler=scheduler, #type: ignore
                                      model_config=model_config,
                                      training_config=training_config,
                                      save_file_dir=save_file_dir,
                                      file_name_stem=save_file_name,
                                      verbose=True)


    print("\nLogging demo of model to tensorboard...")
    log_demo(writer=writer,
             demo_config=demo_config,
             model=model,
             device=device,
             patch_size=patch_size,
             num_epochs=num_epochs,
             verbose=True,
             use_caching=False,
             model_file_path=state_save_file_path)

    # Output time taken

    time_taken = int(time.time() - start_time)
    seconds = time_taken % 60
    minutes = (time_taken // 60) % 60
    hours = time_taken // (60**2)
    print(f"\nTook {hours} hrs {minutes} min in total.")

    writer.close()

# python3 train.py train-config.json5 augmentation-config.json5 model-v3
