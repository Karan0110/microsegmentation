from typing import Union
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

import criterions

def weight_dict_to_tensor(raw_weights : dict,
                          Labels,
                          normalize : bool = False) -> torch.Tensor:
    weights = [None] * len(Labels)
    for label_str in raw_weights:
        index = Labels[label_str].value
        weights[index] = raw_weights[label_str]

    weights = torch.tensor(weights)

    if normalize:
        weights /= weights.sum()

    return weights

def get_criterion(criterion_config : dict,
                  device : torch.device,
                  Labels,
                  verbose : bool = True) -> nn.Module:
    criterion_name = criterion_config['name']

    raw_criterion_params = criterion_config['params']
    new_criterion_params = raw_criterion_params.copy()

    if criterion_name == 'CrossEntropyLoss':
        if 'weight' in raw_criterion_params:
            raw_class_weights = raw_criterion_params['weight']
            new_criterion_params['weight'] = weight_dict_to_tensor(raw_class_weights,
                                                                   Labels=Labels,
                                                                   normalize=True).to(device)

    if criterion_name == 'FocalLoss':
        if 'alpha' in raw_criterion_params:
            raw_weights = raw_criterion_params['alpha']
            new_criterion_params['alpha'] = weight_dict_to_tensor(raw_weights,
                                                                  Labels=Labels,
                                                            normalize=True).to(device)

    if criterion_name == "TverskyLoss":
        new_criterion_params['foreground_label'] = Labels.POLYMERIZED.value

    if hasattr(nn, criterion_name):
        LossFunction = getattr(nn, criterion_name)
    elif hasattr(criterions, criterion_name):
        LossFunction = getattr(criterions, criterion_name)
    else:
        raise ValueError(f"The criterion specified: {criterion_name} is not a valid criterion!")

    criterion : nn.Module = LossFunction(**new_criterion_params)
    criterion = criterion.to(device)

    if verbose:
        print(f"Initialized {criterion_name} criterion")

    return criterion
