from typing import Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

def train_model(model : nn.Module, 
                device : str, 
                train_loader : DataLoader, 
                criterion : nn.Module,  
                optimizer : Optimizer, 
                epoch : int,
                writer : SummaryWriter,
                num_batches : Union[int, None],
                verbose : bool = False) -> None:
    model.to(device)

    running_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if num_batches is not None and batch_idx >= num_batches:
            break
    
        inputs, targets = inputs.to(device), targets.to(device)

        model.train()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
    
        optimizer.step()

        running_loss += loss.item()

    # Log the train loss
    if num_batches is not None:
        num_batches = min(len(train_loader), num_batches)
    else:
        num_batches = len(train_loader)
    train_loss = running_loss / num_batches

    if verbose:
        print(f"Training loss: {running_loss / 10}")

    writer.add_scalar('train loss',
                    train_loss,
                    global_step=epoch)
    writer.flush()

def test_model(model : nn.Module, 
               device : str, 
               patch_size : int,
               tensorboard_demo_config_file_path : Path,
               test_loader : DataLoader, 
               criterion : nn.Module, 
               writer : SummaryWriter,
               epoch : int,
               num_batches : Union[int, None],
               verbose : bool = False) -> None:
    model.to(device)

    test_loss = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            model.eval()

            if num_batches is not None and batch_idx >= num_batches:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()

    if num_batches is not None:
        num_batches = min(len(test_loader), num_batches)
    else:
        num_batches = len(test_loader)
    test_loss /= num_batches

    if verbose:
        print(f"Test loss: {test_loss}")

    writer.add_scalar('test loss',
                      scalar_value=test_loss,
                      global_step=epoch)
    writer.flush()
