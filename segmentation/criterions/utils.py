from typing import Union, List, Tuple
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation.data.labels import Labels

import criterions

def weight_dict_to_tensor(raw_weights : dict,
                          normalize : bool = False) -> torch.Tensor:
    weights = [None] * len(Labels)
    for label_str in raw_weights:
        index = Labels[label_str].value
        weights[index] = raw_weights[label_str]

    weights = torch.tensor(weights)

    if normalize:
        weights /= weights.sum()

    return weights

def _get_criterion(criterion_config : dict,
                  device : torch.device) -> nn.Module:
    criterion_name = criterion_config['name']

    if hasattr(nn, criterion_name):
        LossFunction = getattr(nn, criterion_name)
    elif hasattr(criterions, criterion_name):
        LossFunction = getattr(criterions, criterion_name)
    else:
        raise ValueError(f"The criterion specified: {criterion_name} is not a valid criterion!")

    raw_criterion_params = criterion_config.get('params', {})
    new_criterion_params = raw_criterion_params.copy()

    if issubclass(LossFunction, nn.CrossEntropyLoss):
        if 'weight' in raw_criterion_params:
            raw_class_weights = raw_criterion_params['weight']
            new_criterion_params['weight'] = weight_dict_to_tensor(raw_class_weights,
                                                                   normalize=True).to(device)

    if issubclass(LossFunction, criterions.FocalLoss):
        if 'alpha' in raw_criterion_params:
            raw_weights = raw_criterion_params['alpha']
            new_criterion_params['alpha'] = weight_dict_to_tensor(raw_weights,
                                                                  normalize=True).to(device)

    if issubclass(LossFunction, criterions.TverskyLoss):
        new_criterion_params['foreground_label'] = Labels.POLYMERIZED.value

    criterion : nn.Module = LossFunction(**new_criterion_params)
    criterion = criterion.to(device)

    return criterion

def get_criterions(criterions_config : List[dict],
                   device : torch.device,
                   verbose : bool = False) -> List[dict]:
    criterions : List[dict] = []

    for criterion_config in criterions_config:
        weight = criterion_config.get('weight', 1.0)
        criterion = _get_criterion(criterion_config=criterion_config,
                                   device=device)
        name = criterion_config['name']

        criterions.append({
            'criterion': criterion,
            'weight': weight,
            'name': name,
        })

        if verbose:
            print(f"Initialized {name} criterion with weight {weight}")

    return criterions
