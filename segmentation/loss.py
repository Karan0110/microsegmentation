from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from synthetic_dataset import Labels

class LossNamespace:
    class TverskyLoss(nn.Module):
        def __init__(self,
                     alpha: float,
                     beta : float,
                     eps: float = 1e-6) -> None:
            super().__init__()

            self.alpha = alpha
            # NOTE: Make beta big to penalise missing out poly pixels in ground truth
            self.beta = beta

            self.eps = eps

        # inputs: (B, num_classes, H, W)
        # targets: (B, H, W)
        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            num_classes = inputs.size(1)

            # Convert targets to one-hot encoding
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

            # Ensure inputs and targets_one_hot have the same shape
            if inputs.shape != targets_one_hot.shape:
                raise ValueError(f"Shape mismatch: inputs {inputs.shape}, targets_one_hot {targets_one_hot.shape}")

            # Shape: (B, H, W)
            pred_poly_probs = inputs[:, Labels.POLYMERIZED.value, :, :]
            ground_poly_probs = targets_one_hot[:, Labels.POLYMERIZED.value, :, :]
            assert len(pred_poly_probs.shape) == 3

            # Shape: (B,)
            intersection = (pred_poly_probs * ground_poly_probs).sum(dim=(1,2))
            pred_minus_ground = (pred_poly_probs * (1. - ground_poly_probs)).sum(dim=(1,2))
            ground_minus_pred = ((1. - pred_poly_probs) * ground_poly_probs).sum(dim=(1,2))

            # Shape: (B,)
            tverksy_index = intersection / (intersection + self.alpha * pred_minus_ground + self.beta * ground_minus_pred + self.eps)
            tversky_loss = 1. - tverksy_index

            #Shape: ()
            total_tversky_loss = tversky_loss.sum()

            return total_tversky_loss

    class FocalLoss(nn.Module):
        def __init__(self,
                     alpha: Union[torch.Tensor, None] = None,
                     gamma: float = 2.0) -> None:
            super().__init__()

            self.alpha = alpha
            self.gamma = gamma
            
        # inputs: (B, num_classes, H, W)
        # targets: (B, H, W)
        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            num_classes = inputs.size(1)

            # Convert targets to one-hot encoding
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

            # Ensure inputs and targets_one_hot have the same shape
            if inputs.shape != targets_one_hot.shape:
                raise ValueError(f"Shape mismatch: inputs {inputs.shape}, targets_one_hot {targets_one_hot.shape}")

            # Compute the log probabilities
            # Shape: (B, num_classes, H, W)
            log_p = F.log_softmax(inputs, dim=1)

            # Select the log probabilities corresponding to the targets
            # Shape: (B, H, W)
            log_p = (log_p * targets_one_hot).sum(dim=1)
            cross_entropy_loss = -log_p

            # Apply alpha factor per class
            if self.alpha is not None:
                if self.alpha.dim() == 1:
                    at = self.alpha[targets].view(-1)
                else:
                    at = self.alpha.gather(0, targets.view(-1))
                cross_entropy_loss = cross_entropy_loss * at

            # Compute the focal loss components
            # Shape: (B, H, W)
            pt = torch.exp(-cross_entropy_loss)
            focal_loss = ((1 - pt) ** self.gamma) * cross_entropy_loss

            # Shape: (B,)
            focal_loss = focal_loss.mean(dim=(1,2))

            total_focal_loss = focal_loss.sum()

            return total_focal_loss

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

def get_criterion(loss_config : dict,
                  device : torch.device) -> nn.Module:
    loss_name = loss_config['loss']

    raw_loss_params = loss_config['loss_options'][loss_name]
    new_loss_params = raw_loss_params.copy()

    if loss_name == 'CrossEntropyLoss':
        if 'weight' in raw_loss_params:
            raw_class_weights = raw_loss_params['weight']
            new_loss_params['weight'] = weight_dict_to_tensor(raw_class_weights,
                                                            normalize=True).to(device)

    if loss_name == 'FocalLoss':
        if 'alpha' in raw_loss_params:
            raw_weights = raw_loss_params['alpha']
            new_loss_params['alpha'] = weight_dict_to_tensor(raw_weights,
                                                            normalize=True).to(device)

    if hasattr(nn, loss_name):
        LossFunction = getattr(nn, loss_name)
    elif hasattr(LossNamespace, loss_name):
        LossFunction = getattr(LossNamespace, loss_name)
    else:
        raise ValueError(f"The loss specified: {loss_name} is not a valid loss!")

    criterion = LossFunction(**new_loss_params)

    return criterion
