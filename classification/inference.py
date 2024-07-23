from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms as transforms

from patches import get_image_patches, stitch_image_patches

def get_segmentation(image: np.ndarray,
                     device : str,
                     model : nn.Module, 
                     patch_size : int,
                     batch_size : int = 1) -> Tensor:
    # Convert image to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    transformed_image  = transform(image)
    if not isinstance(transformed_image, Tensor):
        raise ValueError('The transform of the image is not a Tensor')
    image_tensor : Tensor = transformed_image.clone().detach()

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0) 
    _, channels, height, width = image_tensor.shape
    original_image_shape = (1, channels, height, width)

    #Get patches
    patches = get_image_patches(image_tensor, (patch_size, patch_size)).to(device)
    batched_patches = list(torch.split(patches, batch_size, dim=0))

    # Run inference on patches
    model = model.to(device)
    model.eval()
    batched_logits = [None] * len(batched_patches)
    with torch.no_grad():
        for i in range(len(batched_patches)):
            batched_logits[i] = model(batched_patches[i])
    logits = torch.cat(batched_logits, dim=0) #type: ignore

    # Get pixel probabilities
    probabilities = F.softmax(logits, dim=1)[:, 1, :, :].unsqueeze(1)

    # Stitch patches together to get full image segmentation
    probabilities = stitch_image_patches(patches=probabilities,
                                        original_shape=original_image_shape,
                                        patch_size=patch_size)
    probabilities = probabilities.squeeze()

    return probabilities
