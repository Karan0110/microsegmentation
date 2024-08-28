import torch
import torch.nn as nn
import torch.nn.functional as F

from .tversky_loss import TverskyLoss

class DiceLoss(TverskyLoss):
    def __init__(self,
                 foreground_label : int,
                 eps: float = 1e-7) -> None:
        super().__init__(foreground_label=foreground_label, 
                         alpha=0.5, 
                         beta=0.5,
                         eps=eps)
