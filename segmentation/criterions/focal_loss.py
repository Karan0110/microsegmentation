from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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