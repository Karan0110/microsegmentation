# Command line arguments
# save_dir          : Path to model folder
# save_file_name    : Name of save file (.pth)
# patch_size (=256) : A unique ID for the data sample

import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from resnet import ResNet
from epoch import train_model, test_model
from tubulaton_dataloader import get_data_loaders

import tqdm as tqdm

if __name__ == '__main__':
    save_dir = sys.argv[1]
    save_file_name = sys.argv[2]

    if len(sys.argv) <= 3:
        patch_size = 256
    else:
        patch_size = int(sys.argv[3])

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
    train_loader, test_loader = get_data_loaders(patch_size=patch_size)

    # Initialize model, loss function, optimizer, and scheduler
    layers = [2,2,2,2]
    in_channels = 1
    num_classes=2

    model = ResNet(layers=layers, in_channels=in_channels, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training and testing loop
    num_epochs = 30

    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            train_model(model, device, train_loader, criterion, optimizer, epoch)
            pbar_ordered_dict = test_model(model, device, test_loader, criterion)
            scheduler.step()
            
            pbar.update(1)
            pbar.set_postfix(ordered_dict=pbar_ordered_dict)
            
    #Save the model to file
    save_file_path = os.path.join(save_dir, save_file_name)

    torch.save(model.state_dict(),
            f=os.path.join(save_file_path))
    print(f"Saved model to {save_file_path}.")