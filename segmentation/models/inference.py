import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .patches import get_image_patches, stitch_image_patches

from synthetic_dataset import Labels

def get_hard_segmentation(segmentation : np.ndarray,
                          segmentation_threshold : float = 0.5) -> np.ndarray:

    hard_segmentation = np.where(segmentation < segmentation_threshold, 0., 1.)

    return hard_segmentation

def get_segmentation(image: np.ndarray,
                     model : nn.Module, 
                     device : torch.device,
                     patch_size : int,
                     batch_size : int = 1) -> Tensor:
    # Add a channels dimension if not already present
    if len(image.shape) == 2:
        image = image[np.newaxis, :, :]

    # Convert image to tensor
    image_tensor : Tensor
    image_tensor = torch.from_numpy(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0) 
    _, channels, height, width = image_tensor.shape
    original_image_shape = (1, channels, height, width)

    # Get patches
    patches = get_image_patches(image_tensor, (patch_size, patch_size)).to(device)
    batched_patches = list(torch.split(patches, batch_size, dim=0))

    # Prepare model for inference
    model = model.to(device)
    model.eval()

    # Run inference on patches (using specified batch size)
    batched_logits = []
    with torch.no_grad():
        for i in range(len(batched_patches)):
            single_batched_logits = model(batched_patches[i])
            batched_logits.append(single_batched_logits)
    logits = torch.cat(batched_logits, dim=0) 

    # Get pixel probabilities
    probabilities = F.softmax(logits, dim=1)[:, Labels.POLYMERIZED.value, :, :].unsqueeze(1)

    # Stitch patches together to get full image segmentation
    probabilities = stitch_image_patches(patches=probabilities,
                                        original_shape=original_image_shape,
                                        patch_size=patch_size)
    segmentation = probabilities.squeeze()

    #Crop to size of original image (avoid patch overflow)
    segmentation = segmentation[:height, :width]

    return segmentation
