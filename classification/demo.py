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

from unet import load_model
from inference import get_segmentation

import torch
import torch.nn as nn

def plot_demo(image : np.ndarray,
              model : nn.Module,
              config : dict,
              device : str,
              fig : Figure, 
              axs : np.ndarray, 
              row : int = 0) -> None:

    height, width = image.shape
    patch_size = int(config['patch_size'])

    probabilities = get_segmentation(image=image,
                                     device=device,
                                     model=model,
                                     patch_size=patch_size)
    probabilities = probabilities.to('cpu').numpy()
    print(probabilities)

    color_yellow = np.array([255., 255., 0.])
    colored_probabilities = np.tile(probabilities, reps=(3,1,1)) * color_yellow[:, np.newaxis, np.newaxis] 
    colored_probabilities  = colored_probabilities.astype(np.uint8)

    print(colored_probabilities.shape)

    # Convert the image to include an alpha channel
    alpha_channel = probabilities
    alpha_colored_probabilities = np.concatenate((colored_probabilities, alpha_channel[np.newaxis, ...]), axis=0)

    # Reshape images to have RGB(A) at axis 2
    colored_probabilities = np.transpose(colored_probabilities, axes=(1,2,0))
    alpha_colored_probabilities = np.transpose(alpha_colored_probabilities, axes=(1,2,0))

    # Draw image with probabilities overlaid
    axs[row,0].imshow(image, cmap='gray')
    # axs[row,0].imshow(alpha_colored_probabilities)
    axs[row,0].axis('off')

    # Draw probabilities on their own
    axs[row,1].imshow(probabilities, cmap='gray')
    axs[row,1].axis('off')
    
if __name__ == '__main__':
    if len(sys.argv) not in [3,4]:
        print("Invalid command line arguments!")
        print("Correct usage:")
        print(f"python3 {sys.argv[0]} [model_file_path] [image_path] <cutoff>")
        print("Where [model_file_path] can be set to \"default\" to use /Users/karan/Microtubules/Models/model.pth")
        print("[image_path] is either a file or a dir of images")
        print("<cutoff> says max number of images to demo (Only used when [image_path] is dir)")
        exit(1)

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
    config : dict
    model, config = load_model(model_file_path=model_file_path,
                               device=device)
    print("Loaded trained U-Net model.")

    if image_path.is_file():
        fig, axs = plt.subplots(1, 2, figsize=(10, 2))
        axs = axs.reshape((1,2))
        image_file_path = image_path
        image = np.array(Image.open(image_file_path).convert('L'))
        plot_demo(image=image,
                  model=model,
                  config=config,
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
            plot_demo(image=image,
                      model=model,
                      config=config,
                      device=device,
                      fig=fig,
                      axs=axs,
                      row=row)
        plt.show()

# python3 demo.py /Users/karan/Microtubules/Models/model-e41.pth /Users/karan/MTData/SimulatedData/Images/image-430.png
