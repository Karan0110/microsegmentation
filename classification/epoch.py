import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer

def train_model(model : nn.Module, 
                device : str, 
                train_loader : DataLoader, 
                criterion : nn.Module,  
                optimizer : Optimizer, 
                epoch : int) -> None:
    model.to(device)
    model.train()
    running_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
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
               verbose : bool = False) -> dict:
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets.max(1)[1]).sum().item()
            
    accuracy = 100. * correct / total
    
    if verbose:
        print(f"Test loss: {test_loss}")
        print(f"Accuracy: {accuracy}")
        
    return {
        'Test Loss': test_loss,
        'Accuracy': accuracy,
    }
