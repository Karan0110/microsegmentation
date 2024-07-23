from typing import Tuple

import numpy as np

from PIL import Image

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms as transforms

from resnet import ResNet

#Returns tensor of shape (num_patches, channels, height, width)
#The patches are ordered lexicographically, e.g.
# 0 1 2
# 3 4 5
def get_image_patches(image : Tensor, 
                      patch_size : Tuple[int,int]) -> Tensor:
    # batch, channels, height, width
    b, c, h, w = image.shape
    patch_h, patch_w = patch_size
    
    # Apply circular padding
    pad_h = (patch_h - (h % patch_h)) % patch_h
    pad_w = (patch_w - (w % patch_w)) % patch_w

    padded_image = None
    if pad_h > 0 or pad_w > 0:
        padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode='circular')
    else:
        padded_image = image

    # Unfold to extract patches
    patches = padded_image.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    
    # Reshape tensor
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, c, patch_h, patch_w)
    
    return patches

def get_patch_probabilities(image: np.ndarray,
                            device : str,
                            model : nn.Module, 
                            patch_size : int) -> Tensor:
    # Load image into appropriate tensor shape
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Add batch dimension
    transformed_image  = transform(image)
    if not isinstance(transformed_image, Tensor):
        raise ValueError('The transform of the image is not a Tensor')
    image_tensor : Tensor = transformed_image.clone().detach()
    image_tensor = image_tensor.unsqueeze(0) 
    _, _, height, width = image_tensor.shape

    #Get patches
    patches = get_image_patches(image_tensor, (patch_size, patch_size)).to(device)

    # Run inference on each patch and collect probabilities
    with torch.no_grad():
        logits = model(patches)
    probabilities : Tensor = F.softmax(logits, dim=1)[:, 0].to('cpu').numpy()

    return probabilities

