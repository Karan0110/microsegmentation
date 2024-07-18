# Command line arguments
# dataset_dir       : Path to dataset folder
# save_dir          : Path to model folder
# save_file_name    : Name of save file (without .pth)
# patch_size (=256) : A unique ID for the data sample

import os
import sys
from pathlib import Path
from pprint import pprint
import json5

import torch
import torch.nn as nn
import torch.optim as optim

from resnet import ResNet
from epoch import train_model, test_model
from tubulaton_dataset import get_data_loaders

from tqdm import tqdm

if __name__ == '__main__':
    if len(sys.argv) not in [2,3]:
        print("Too few / many command-line arguments!")
        print("Correct usage of program:")
        print(f"python3 {sys.argv[0]} [config_file_path] <save_file_name>")
        print("Where default name for model save file is \"model\"")
        exit(1)

    config_file_path = Path(sys.argv[1])
    save_file_name = sys.argv[2] if len(sys.argv) >= 3 else "model"

    with config_file_path.open('r') as file:
        config = json5.load(file)

    if not isinstance(config, dict):
        raise TypeError("JSON5 config file is of invalid format! It should be a dictionary")

    dataset_dir = Path(config['dataset_dir'])
    save_file_dir = Path(config['save_file_dir'])
    os.makedirs(save_file_dir, exist_ok=True)

    save_file_path = save_file_dir / f"{save_file_name}.pth"

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
    train_loader, test_loader = get_data_loaders(patch_size=patch_size,
                                                 base_dir=dataset_dir,
                                                 transform_config=transform_config)

    # Initialize model, loss function, optimizer, and scheduler
    model_config = config['model']
    layers = model_config['layers']
    in_channels = model_config['in_channels']
    num_classes = model_config['num_classes']

    model = ResNet(layers=layers, in_channels=in_channels, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_config = config['optimizer']
    optimizer = optim.SGD(model.parameters(), **optimizer_config)

    scheduler_config = config['scheduler']
    scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_config)

    # Training and testing loop
    num_epochs = config['num_epochs']

    for epoch in range(num_epochs):
        train_model(model, device, train_loader, criterion, optimizer, epoch)
        progress_report = test_model(model, device, test_loader, criterion)
        scheduler.step()

        for key in progress_report:
            print(f"{key}: {progress_report[key]}")
            
    #Save the model to file
    model_state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': num_epochs,

        'model_params': {
            'patch_size': patch_size,
            'layers': layers,
            'in_channels': in_channels,
            'num_classes': num_classes,
        }
    }

    torch.save(model_state,
               f=save_file_path)

    print(f"Saved model to {save_file_path}.")
