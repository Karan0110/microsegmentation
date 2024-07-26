import sys
import os
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from PIL import Image

from resnet import ResNet
from load_resnet import load_model
from patches_resnet import get_patch_probabilities

import torch
import torch.nn as nn

def draw_patches(ax : Axes,
                 probabilities : list,
                 height_in_patches : int,
                 width_in_patches : int,
                 patch_size : int,
                 alpha : float = 1.) -> None:
    
    for i in range(height_in_patches):
        for j in range(width_in_patches):
            index = 4*i + j

            #Probability patch is polymerised (control)
            probability = probabilities[index]

            color = matplotlib.colormaps.get_cmap('inferno')(probability) # type: ignore 
            color_with_alpha = color[:3] + (alpha,)
            square = Rectangle((j * patch_size, i * patch_size), patch_size, patch_size, 
                                   facecolor=color_with_alpha, edgecolor='green')

            ax.add_patch(square)


def plot_demo(detailed : bool,
              image : np.ndarray,
              model : nn.Module,
              model_params : dict,
              device : str,
              fig : Figure, 
              axs : np.ndarray, 
              row : int = 0) -> None:
    height, width = image.shape
    patch_size = int(model_params['patch_size'])

    probabilities = get_patch_probabilities(image=image,
                                            device=device,
                                            model=model,
                                            patch_size=patch_size)

    #Ceiling division
    height_in_patches = (height + patch_size - 1) // patch_size
    width_in_patches = (width + patch_size - 1) // patch_size

    # Draw image with probabilities overlaid
    axs[row,0].imshow(image, cmap='gray')
    axs[row,0].axis('off')
    draw_patches(probabilities=probabilities.tolist(),
                 ax=axs[row,0],
                 width_in_patches=width_in_patches,
                 height_in_patches=height_in_patches,
                 patch_size=patch_size,
                 alpha=0.7)

    # Draw probabilities on their own
    axs[row,1].axis('off')
    axs[row,1].set_xlim(0, width_in_patches * patch_size)
    axs[row,1].set_ylim(height_in_patches * patch_size, 0)

    draw_patches(probabilities=probabilities.tolist(),
                 ax=axs[row,1],
                 width_in_patches=width_in_patches,
                 height_in_patches=height_in_patches,
                 patch_size=patch_size,
                 alpha=1.)
    
    if detailed:
        #Draw Colorbar
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        axs[row+1,0].imshow(gradient, aspect='auto', cmap='inferno', extent=(0, 1, 0.49, 0.51))
        axs[row+1,0].axis('off')

        #Draw histogram
        bins = np.histogram_bin_edges(probabilities, bins='auto').tolist()
        axs[row+1,1].hist(probabilities, bins=bins, edgecolor='black')
        axs[row+1,1].set_xlabel('Patch Polymerisation Probability')
        axs[row+1,1].set_ylabel('Frequency')

if __name__ == '__main__':
    if sys.argv[1] == 'default':
        model_file_path = Path("/Users/karan/Microtubules/Models/model.pth")
    else:
        model_file_path = Path(sys.argv[1])
    image_path = Path(sys.argv[2])
    if len(sys.argv) >= 4:
        cutoff = int(sys.argv[3])
    else:
        cutoff = None

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


    if image_path.is_file():
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        image_file_path = image_path
        image = np.array(Image.open(image_file_path).convert('L'))
        plot_demo(detailed=True,
                  image=image,
                  model=model,
                  model_params=model_params,
                  device=device,
                  fig=fig,
                  axs=axs,
                  row=0)
        plt.show()
    else:
        image_dir_path = image_path
        image_file_paths = [image_file_path for image_file_path in image_dir_path.iterdir() if image_file_path.name.endswith('.png')]
        random.shuffle(image_file_paths)
        if cutoff is None:
            cutoff = len(image_file_paths)
        image_file_paths = image_file_paths[:cutoff]

        images = [np.array(Image.open(image_file_path).convert('L')) for image_file_path in image_file_paths]

        fig, axs = plt.subplots(cutoff, 2, figsize=(10, 5))

        for row,image in enumerate(images):
            plot_demo(detailed=False,
                      image=image,
                      model=model,
                      model_params=model_params,
                      device=device,
                      fig=fig,
                      axs=axs,
                      row=row)
        plt.show()
