import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
from matplotlib.axes import Axes

from PIL import Image

from resnet import ResNet
from load_model import load_model
from patches import get_patch_probabilities

import torch
import torch.nn as nn

def draw_patches(ax : Axes,
                 probabilities : list,
                 height_in_patches : int,
                 width_in_patches : int,
                 patch_size : int,
                 alpha : float = 1.) -> None:
    ax.axis('off')
    for i in range(height_in_patches):
        for j in range(width_in_patches):
            index = 4*i + j

            #Probability patch is polymerised (control)
            probability = probabilities[index]

            color = matplotlib.colormaps.get_cmap('inferno')(probability) # type: ignore 
            color_with_alpha = color[:3] + (alpha,)
            square = Rectangle((j * patch_size, i * patch_size), patch_size, patch_size, 
                                   facecolor=color_with_alpha, edgecolor='none')

            ax.add_patch(square)


if __name__ == '__main__':
    model_file_path = Path(sys.argv[1])
    image_file_path = Path(sys.argv[2])

    # General device handling: Check for CUDA/GPU, else fallback to CPU
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model : nn.Module
    model_params : dict
    model, model_params = load_model(model_file_path=model_file_path,
                                     device=device)
    print("Loaded trained ResNet model.")

    patch_size = int(model_params['patch_size'])

    image = np.array(Image.open(image_file_path).convert('L'))
    height, width = image.shape

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    probabilities = get_patch_probabilities(image=image,
                                            device=device,
                                            model=model,
                                            patch_size=patch_size)

    #Ceiling division
    height_in_patches = (height + patch_size - 1) // patch_size
    width_in_patches = (width + patch_size - 1) // patch_size

    # Draw image with probabilities overlaid
    axs[0,0].imshow(image, cmap='gray')
    draw_patches(probabilities=probabilities.tolist(),
                 ax=axs[0,0],
                 width_in_patches=width_in_patches,
                 height_in_patches=height_in_patches,
                 patch_size=patch_size,
                 alpha=0.7)

    # Draw probabilities on their own
    axs[0,1].axis('off')
    draw_patches(probabilities=probabilities.tolist(),
                 ax=axs[0,1],
                 width_in_patches=width_in_patches,
                 height_in_patches=height_in_patches,
                 patch_size=patch_size,
                 alpha=1.)
    
    #Draw Colourbar
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    axs[1,0].imshow(gradient, aspect='auto', cmap='inferno', extent=(0, 1, 0.49, 0.51))
    axs[1,0].axis('off')

    #Draw histogram
    bins = np.histogram_bin_edges(probabilities, bins='auto').tolist()
    axs[1,1].hist(probabilities, bins=bins, edgecolor='black')
    axs[1,1].set_xlabel('Patch Polymerisation Probability')
    axs[1,1].set_ylabel('Frequency')

    plt.show()
