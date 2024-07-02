# Command line arguments
# dataset_dir       : Path to dataset folder
# save_dir          : Path to model folder
# save_file_name    : Name of save file (without .pth)
# patch_size (=256) : A unique ID for the data sample

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from resnet import ResNet
from epoch import train_model, test_model
from tubulaton_dataloader import get_data_loaders

from tqdm import tqdm

if __name__ == '__main__':
    dataset_dir = Path(sys.argv[1])
    save_file_dir = Path(sys.argv[2])
    save_file_name = f"{sys.argv[3]}.pth"
    patch_size = int(sys.argv[4])

    save_file_path = save_file_dir / save_file_name

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
    train_loader, test_loader = get_data_loaders(patch_size=patch_size,
                                                 base_dir=dataset_dir)

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

    with tqdm(total=num_epochs, ncols=100) as pbar:
        for epoch in range(num_epochs):
            train_model(model, device, train_loader, criterion, optimizer, epoch)
            pbar_ordered_dict = test_model(model, device, test_loader, criterion)
            scheduler.step()
            
            pbar.update(1)
            pbar.set_postfix(ordered_dict=pbar_ordered_dict)
            
    #Save the model to file
    model_state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),

        'epoch': num_epochs,
        'model_params': {
            'layers': layers,
            'in_channels': in_channels,
            'num_clases': num_classes,
        }
    }

    torch.save(model_state,
            f=os.path.join(save_file_path))
    print(f"Saved model to {save_file_path}.")