from typing import Tuple, Union, List
from pathlib import Path
import time
import os
import hashlib

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms as transforms

from patches import get_image_patches, stitch_image_patches

from synthetic_dataset import Labels

# For caching segmentation results
# Feed in the input image and the model file
def get_cache_file_path(model_file_path : Path,
                        image_file_path : Path) -> Path:
    hasher = hashlib.sha256()

    with model_file_path.open('rb') as f:
        hasher.update(f.read())
    with image_file_path.open('rb') as f:
        hasher.update(f.read())

    return Path(f".segmentation_cache/cache-{hasher.hexdigest()}.npy")

def get_hard_segmentation(segmentation : np.ndarray,
                          segmentation_threshold : float = 0.5,
                          verbose : bool = True) -> np.ndarray:
    hard_segmentation = np.where(segmentation < segmentation_threshold, 0., 1.)

    return hard_segmentation

def get_segmentation(image: np.ndarray,
                     device : torch.device,
                     model : nn.Module, 
                     patch_size : int,
                     batch_size : int = 1,
                     use_caching : bool = True,
                     model_file_path : Union[Path, None] = None,
                     image_file_path : Union[Path, None] = None,
                     verbose : bool = False) -> np.ndarray:
    if use_caching:
        if model_file_path is None:
            raise ValueError("Cannot use caching without being provided model file path!")
        if image_file_path is None:
            raise ValueError("Cannot use caching without being provided image file path!")
        
        cache_file_path = get_cache_file_path(model_file_path, image_file_path)

        if cache_file_path.exists():
            segmentation = np.load(cache_file_path)
            if verbose:
                print(f"Loaded segmentation from cache: {cache_file_path}")
            return segmentation
        else:
            if verbose:
                print("No cache exists for this model/image pair so manually computing segmentation...")

    start_time = time.time()
    
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
    probabilities = F.softmax(logits, dim=1)[:, Labels.POLYMERIZED.value, :, :].unsqueeze(1)

    # Stitch patches together to get full image segmentation
    probabilities = stitch_image_patches(patches=probabilities,
                                        original_shape=original_image_shape,
                                        patch_size=patch_size)
    segmentation = probabilities.squeeze()
    segmentation = segmentation.to('cpu').numpy()

    #Crop to size of original image (avoid patch overflow)
    segmentation = segmentation[:height, :width]

    time_taken = time.time() - start_time    

    if verbose:
        print(f"Ran inference in {time_taken} seconds.")

    if use_caching:
        os.makedirs(cache_file_path.parent, exist_ok=True) #type: ignore
        np.save(cache_file_path, segmentation) #type: ignore

        if verbose:
            print(f"Saved to cache at: {cache_file_path}") #type: ignore

    return segmentation
