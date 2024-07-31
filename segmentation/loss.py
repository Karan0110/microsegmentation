from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from synthetic_dataset import Labels

class LossNamespace:
    class FocalLoss(nn.Module):
        def __init__(self, 
                    alpha : Union[torch.Tensor, None] = None,
                    gamma : float = 2.0, 
                    reduction : str = 'mean') -> None:
            super().__init__()

            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            num_classes = inputs.size(1) 

            # Convert targets to one-hot encoding
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
            
            # Compute the cross entropy loss
            log_p = F.log_softmax(inputs, dim=1)
            log_p = log_p.gather(1, targets_one_hot.argmax(dim=1, keepdim=True))
            cross_entropy_loss = -log_p.squeeze(1)

            # Apply alpha factor per class
            if self.alpha is not None:
                at = self.alpha.gather(0, targets_one_hot.argmax(dim=1))
                cross_entropy_loss = cross_entropy_loss * at

            # Compute the focal loss components
            pt = torch.exp(-cross_entropy_loss)
            focal_loss = (1 - pt) ** self.gamma * cross_entropy_loss
            
            # Apply reduction method
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss

def get_criterion(loss_config : dict,
                  device : torch.device) -> nn.Module:
    loss_name = loss_config['name']

    raw_loss_params = loss_config['params']
    new_loss_params = raw_loss_params.copy()

    if 'weight' in raw_loss_params:
        raw_class_weights = raw_loss_params['weight']
        class_weights = [None] * len(Labels)
        for label_str in raw_class_weights:
            index = Labels[label_str].value
            class_weights[index] = raw_class_weights[label_str]
        new_loss_params['weight'] = torch.tensor(class_weights).to(device)

    if hasattr(nn, loss_name):
        LossFunction = getattr(nn, loss_name)
    elif hasattr(LossNamespace, loss_name):
        LossFunction = getattr(LossNamespace, loss_name)
    else:
        raise ValueError(f"The loss specified: {loss_name} is not a valid loss!")

    criterion = LossFunction(**new_loss_params)

    return criterion