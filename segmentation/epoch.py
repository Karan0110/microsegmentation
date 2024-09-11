from typing import Union, List, Tuple, Dict
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
                criterions : List[dict],
                optimizer : Optimizer, 
                epoch : int,
                writer : SummaryWriter,
                verbose : bool = False) -> None:
    model.to(device)

    running_loss = 0.0
    running_count : int = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        model.train()

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = torch.tensor(0.0).to(device)
        for criterion in criterions:
            if criterion['weight'] == 0.0:
                continue
            loss += criterion['criterion'](outputs, targets) * criterion['weight']
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

def eval_model(model : nn.Module, 
               device : torch.device, 
               eval_loader : DataLoader, 
               criterions : List[dict],
               writer : SummaryWriter,
               epoch : int,
               verbose : bool = False) -> float:
    model.to(device)

    running_losses = {criterion['name']: 0.0 for criterion in criterions}

    running_count = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(eval_loader):
            model.eval()

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            for criterion in criterions:
                loss = criterion['criterion'](outputs, targets)
                running_losses[criterion['name']] += loss.item()

            running_count += inputs.size(0)

    eval_losses = {name: loss/running_count for name, loss in running_losses.items()}

    for name in eval_losses:
        running_losses[name] /= running_count

        writer.add_scalar(f'loss/{name}',
                        scalar_value=running_losses[name],
                        global_step=epoch)
    writer.flush()

    eval_loss = 0.0
    for criterion in criterions:
        weight = criterion['weight']
        loss = eval_losses[criterion['name']]

        eval_loss += weight * loss

    return eval_loss
