import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    def __init__(self,
                    alpha: float,
                    beta : float,
                    foreground_label : int,
                    eps: float = 1e-6) -> None:
        super().__init__()

        self.alpha = alpha
        # NOTE: Make beta big to penalise missing out poly pixels in ground truth
        self.beta = beta

        self.eps = eps

        self.foreground_label = foreground_label

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
        pred_poly_probs = inputs[:, self.foreground_label, :, :]
        ground_poly_probs = targets_one_hot[:, self.foreground_label, :, :]
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