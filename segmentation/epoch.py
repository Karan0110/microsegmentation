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
    running_count : int = 0

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
        running_count += inputs.size(0)

    # Log the train loss
    train_loss = running_loss / running_count

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

    running_count = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            model.eval()

            if num_batches is not None and batch_idx >= num_batches:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            running_count += inputs.size(0)


    test_loss = running_loss / running_count

    writer.add_scalar('loss/test',
                      scalar_value=test_loss,
                      global_step=epoch)
    writer.flush()

    return test_loss
