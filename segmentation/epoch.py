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
                device : torch.device, 
                train_loader : DataLoader, 
                criterion : nn.Module,  
                optimizer : Optimizer, 
                epoch : int,
                writer : SummaryWriter,
                num_batches : Union[int, None],
                verbose : bool = False) -> None:
    model.to(device)

    running_loss = 0.0
    
    batch_size : Union[int, None] = None

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if num_batches is not None and batch_idx >= num_batches:
            break
    
        inputs, targets = inputs.to(device), targets.to(device)

        if batch_size is None:
            batch_size = inputs.size(0)
        elif batch_size != inputs.size(0):
            raise ValueError(f"Inconsistent batch sizes in training data! ({batch_size} and {inputs.size(0)})")

        model.train()

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
    
        optimizer.step()

        running_loss += loss.item()

    if batch_size is None:
        raise ValueError(f"There was no training data to infer batch_size from!")

    # Log the train loss
    if num_batches is not None:
        num_batches = min(len(train_loader), num_batches)
    else:
        num_batches = len(train_loader)
    train_loss = running_loss / (num_batches * batch_size)

    writer.add_scalar('loss/train',
                      train_loss,
                      global_step=epoch)
    writer.flush()

def test_model(model : nn.Module, 
               device : torch.device, 
               test_loader : DataLoader, 
               criterion : nn.Module, 
               writer : SummaryWriter,
               epoch : int,
               num_batches : Union[int, None],
               verbose : bool = False) -> float:
    model.to(device)

    running_loss = 0.0

    batch_size : Union[int, None] = None

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            model.eval()

            if batch_size is None:
                batch_size = inputs.size(0)
            elif batch_size != inputs.size(0):
                raise ValueError(f"Inconsistent batch sizes in training data! ({batch_size} and {inputs.size(0)})")

            if num_batches is not None and batch_idx >= num_batches:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

    if batch_size is None:
        raise ValueError(f"There was no testing data to infer batch_size from!")

    if num_batches is not None:
        num_batches = min(len(test_loader), num_batches)
    else:
        num_batches = len(test_loader)
    test_loss = running_loss / (num_batches * batch_size)

    writer.add_scalar('loss/test',
                      scalar_value=test_loss,
                      global_step=epoch)
    writer.flush()

    return test_loss
