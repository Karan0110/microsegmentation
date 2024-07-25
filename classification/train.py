# Command line arguments
# dataset_dir       : Path to dataset folder
# save_dir          : Path to model folder
# save_file_name    : Name of save file (without .pth)
# patch_size (=256) : A unique ID for the data sample

import os
import sys
from pathlib import Path
import json5

import torch
import torch.nn as nn
import torch.optim as optim

from unet import UNet
from epoch import train_model, test_model
from synthetic_dataset import get_data_loaders

if __name__ == '__main__':
    if len(sys.argv) not in [2,3]:
        print("Too few / many command-line arguments!")
        print("Correct usage of program:")
        print(f"python3 {sys.argv[0]} [config_file_path] <save_file_name>")
        print("Where default name for model save file is \"model\"")
        exit(1)

    config_file_path = Path(sys.argv[1])
    if config_file_path.suffix != '.json5':
        print(f"Invalid config file path {config_file_path}")
        print(f"Config file must be .json5")
        exit(1)

    save_file_name = sys.argv[2] if len(sys.argv) >= 3 else "model"

    with config_file_path.open('r') as file:
        config = json5.load(file)
    if not isinstance(config, dict):
        raise TypeError(f"JSON5 config file {config_file_path} is of invalid format! It should be a dictionary")

    dataset_dir = Path(config['dataset_dir'])
    save_file_dir = Path(config['save_file_dir'])
    os.makedirs(save_file_dir, exist_ok=True)

    patch_size = config['patch_size']

    # General device handling: Check for CUDA/GPU, else fallback to CPU
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Data
    transform_config = config['transforms']
    color_to_label = config['color_to_label']
    color_to_label = {int(key) : value for key, value in color_to_label.items()}
    batch_size = config['batch_size']
    train_test_split = config['train_test_split']
    batches_per_epoch = config['batches_per_epoch']
    batches_per_test = config['batches_per_test']

    train_loader, test_loader = get_data_loaders(patch_size=patch_size,
                                                 base_dir=dataset_dir,
                                                 transform_config=transform_config,
                                                 color_to_label=color_to_label,
                                                 batch_size=batch_size,
                                                 train_test_split=train_test_split)

    # Initialize model, loss function, optimizer, and scheduler
    model_config = config['model']
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

    criterion = nn.CrossEntropyLoss()

    optimizer_config = config['optimizer']
    optimizer = optim.SGD(model.parameters(), **optimizer_config)

    scheduler_config = config['scheduler']
    scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_config)

    # Training and testing loop
    num_epochs = config['num_epochs']

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}")
        print('-'*15 + '\n')

        train_model(model=model,
                    device=device,
                    train_loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    epoch=epoch,
                    num_batches=batches_per_epoch)
        progress_report = test_model(model=model, 
                                     device=device, 
                                     test_loader=test_loader, 
                                     criterion=criterion,
                                     num_batches=batches_per_test)
        scheduler.step()

        for key in progress_report:
            print(f"{key}: {progress_report[key]}")
            
        #Save the model to file (each epoch)
        model_state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': num_epochs,

            'config': config,
        }

        save_file_path = save_file_dir / f"{save_file_name}-e{epoch}.pth"

        torch.save(model_state,
                   f=save_file_path)

        print(f"Saved model to {save_file_path}.")

# python3 train.py /Users/karan/Microtubules/classification/config.json5
