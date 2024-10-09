from pathlib import Path
import os
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .patches import get_image_patches, stitch_image_patches

from global_utils.hashing import hash_files
from global_utils import load_grayscale_image

from segmentation.data.synthetic_dataset import Labels

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

def get_segmentation_file_path(save_dir : Path,
                               model_file_path : Path,
                               image_file_path : Path) -> Path:
    file_path = save_dir / f"segmentation_{hash(str(model_file_path.absolute()))}_{hash(str(image_file_path.absolute()))}.npy"
        
    return file_path

# Load inference from .inference_save_files or if it doesn't exist yet, run the inference 
# and save the result for next time
def query_inference(model : nn.Module,
                    device : torch.device,
                    image_file_path : Path,
                    model_file_path : Path,
                    patch_size : int,
                    save_dir : Path,
                    overwrite_save_file : bool,
                    verbose : bool,
                    ) -> np.ndarray:
    image = load_grayscale_image(image_file_path=image_file_path)

    file_path = get_segmentation_file_path(save_dir=save_dir,
                                           model_file_path=model_file_path,
                                           image_file_path=image_file_path)

    if file_path.exists() and (not overwrite_save_file):
        if verbose:
            print(f"Segmentation save file found: {file_path}")
            print(f"Loading segmentation from file...")
        segmentation = np.load(file_path) 
        segmentation = segmentation.astype(np.float32) / 255.
    else:
        segmentation = get_segmentation(image=image,
                                        model=model,
                                        device=device,
                                        patch_size=patch_size).to('cpu').numpy()

        if verbose:
            if file_path.exists():
                print(f"Overwriting save file: {file_path}")
            else:
                print(f"Saving result of inference to cache file: {file_path}")

        segmentation_8bit = (segmentation * 255).astype(np.uint8)
        os.makedirs(file_path.parent, exist_ok=True)
        np.save(file_path, segmentation_8bit)

    return segmentation
