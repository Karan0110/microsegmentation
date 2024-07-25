from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer

def train_model(model : nn.Module, 
                device : str, 
                train_loader : DataLoader, 
                criterion : nn.Module,  
                optimizer : Optimizer, 
                epoch : int,
                num_batches : Union[int, None] = None) -> None:
    model.to(device)
    model.train()

    running_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if num_batches is not None and batch_idx >= num_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
    
        optimizer.step()

        running_loss += loss.item()

def test_model(model : nn.Module, 
               device : str, 
               test_loader : DataLoader, 
               criterion : nn.Module, 
               verbose : bool = False,
               num_batches : Union[int, None] = None) -> dict:
    model.to(device)
    model.eval()

    test_loss = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            
    if verbose:
        print(f"Test loss: {test_loss}")
        
    return {
        'Test Loss': test_loss,
    }
