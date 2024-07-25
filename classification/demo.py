import sys
import os
from pathlib import Path
import random
from typing import Tuple, Union
import json5
import time

import numpy as np
import matplotlib.pyplot as plt

import cv2
from PIL import Image

from unet import load_model
from inference import get_segmentation

import torch
import torch.nn as nn

def get_num_cols_to_show(columns : dict,
                         overlays : list) -> int:
    count = 0
    
    for col_name in columns:
        col_info = columns[col_name]
        if col_info['show']:
            count += 1

    for overlay_info in overlays:
        if overlay_info['show']:
            count += 1

    return count 

def get_colored_image(image : np.ndarray,
                      image_color : np.ndarray) -> np.ndarray:
    colored_image = np.empty((*image.shape, 3))
    colored_image = image_color[np.newaxis, np.newaxis, :] * image[..., np.newaxis] + 1. * (1 - image[..., np.newaxis])
    # Deal with float rounding errors
    colored_image = np.clip(colored_image, 0., 1.) 

    return colored_image

def load_grayscale_image(image_file_path : Path) -> np.ndarray:
    image = np.array(Image.open(image_file_path).convert('L'))
    image = image.astype(np.float32) / 255.

    return image

def get_hard_segmentation(segmentation : np.ndarray,
                          segmentation_threshold : Union[str, float],
                          verbose : bool = True) -> Tuple[float, np.ndarray]:
    if isinstance(segmentation_threshold, str): #segmentation_threshold == 'otsu':
        if verbose:
            print("Calculating Otsu threshold for segmentation...")

        cv2_segmentation = (segmentation * 255.).astype(np.uint8)
        segmentation_threshold, hard_segmentation = cv2.threshold(cv2_segmentation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        segmentation_threshold /= 255. #type: ignore
        hard_segmentation = hard_segmentation.astype(np.float32) / 255.

        if verbose:
            print(f"Otsu threshold for segmentation: ({segmentation_threshold})")
    else:
        hard_segmentation = np.where(segmentation < segmentation_threshold, 0., 1.)

    return segmentation_threshold, hard_segmentation

def pad_image_like(image1 : np.ndarray, image2 : np.ndarray) -> np.ndarray:
    padded_image = np.pad(array=image1, 
                          pad_width=((0, image2.shape[0] - image1.shape[0]), (0, image2.shape[1] - image1.shape[1])), 
                          mode='constant', 
                          constant_values=0.)
    return padded_image

def set_axes_mode(axs : np.ndarray,
                  axes_mode : str) -> None:
    for j in range(axs.shape[1]):
        if axes_mode == 'off':
            axs[row, j].axis('off')
        elif axes_mode == 'no_tickers':
            axs[row, j].set_xticks([])
            axs[row, j].set_yticks([])
        elif axes_mode == 'full':
            pass
        else:
            raise ValueError(f"Invalid axes_mode: {axes_mode} (Valid options are: 'off', 'no_tickers', 'full')")

def plot_demo_row(image_file_path : Path,
                  label_file_path : Union[Path, None],
                  model : nn.Module,
                  device : str,
                  patch_size : int,
                  demo_config : dict,
                  axs : np.ndarray, 
                  row : int) -> None:
    segmentation_cache_dir = Path(demo_config["segmentation_cache_dir"])
    axes_mode = demo_config['axes_mode']

    columns = demo_config['columns']
    overlays = demo_config['overlays']

    set_title = (row == 0)

    images = {}

    images['image'] = load_grayscale_image(image_file_path=image_file_path)

    if label_file_path is not None:
        images['label'] = load_grayscale_image(image_file_path=label_file_path)

    model_file_path = Path(demo_config['model_file_path'])
    model_name = model_file_path.stem
    segmentation_cache_file_path = segmentation_cache_dir / f'{model_name}/{image_file_path.stem}.npy'
    images['segmentation'] = get_segmentation(image=images['image'],
                                    device=device,
                                    model=model,
                                    patch_size=patch_size,
                                    cache_file_path=segmentation_cache_file_path,
                                    verbose=True)

    config_segmentation_threshold : Union[str, float] = columns['hard_segmentation']['segmentation_threshold']
    segmentation_threshold : float
    segmentation_threshold, images['hard_segmentation'] = get_hard_segmentation(segmentation=images['segmentation'],
                                                                                segmentation_threshold=config_segmentation_threshold)
    # Pad image to be whole number of patches (and align with segmentation)
    images['image'] = pad_image_like(images['image'], images['segmentation'])
    if label_file_path is not None:
        images['label'] = pad_image_like(images['label'], images['segmentation'])

    colored_images = {}

    ax_index = 0
    for col_name in columns:
        col_info = columns[col_name]

        do_show = col_info['show']
        if not do_show:
            continue

        color = np.array(col_info['color'])
        title = col_info['title']

        grayscale_image = images[col_name]
        colored_image = get_colored_image(image=grayscale_image,
                                          image_color=color)

        colored_images[col_name] = colored_image

        axs[row, ax_index].imshow(colored_image)

        if set_title:
            axs[row, ax_index].set_title(title)

        ax_index += 1

    for overlay_info in overlays:
        if not overlay_info['show']:
            continue

        image_1_name = overlay_info['image_1']
        image_2_name = overlay_info['image_2']

        image_1 = colored_images[image_1_name]
        image_2 = colored_images[image_2_name]

        interpolation = overlay_info['interpolation']

        overlaid_image = cv2.addWeighted(image_1, interpolation, 
                                         image_2, 1. - interpolation, 
                                         0)

        axs[row, ax_index].imshow(overlaid_image)

        if set_title:
            title = f"{image_1_name} overlaid with {image_2_name}"
            axs[row, ax_index].set_title(title)

        ax_index += 1

    set_axes_mode(axs=axs,
                  axes_mode=axes_mode)

if __name__ == '__main__':
    if len(sys.argv) not in [2,]:
        print("Invalid command line arguments!")
        print("Correct usage:")
        print(f"python3 {sys.argv[0]} [demo_config_file_path]")
        exit(1)
    
    demo_config_file_path = Path(sys.argv[1])
    with demo_config_file_path.open('r') as demo_config_file:
        demo_config = json5.load(demo_config_file)
    if not isinstance(demo_config, dict):
        print(f"Invalid demo config file {demo_config_file_path}!")
        print("Must be JSON5 file encoding a dictionary")
        exit(1)
    print(f"Loaded config file: {demo_config_file_path}")

    model_file_path = Path(demo_config['model_file_path'])

    image_label_pair_file_paths = demo_config['image_label_pair_file_paths']

    if demo_config['shuffle_data']:
        random.shuffle(image_label_pair_file_paths)
    
    max_num_demos = demo_config['max_num_demos']
    if max_num_demos is not None:
        image_label_pair_file_paths = image_label_pair_file_paths[:max_num_demos]

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
    patch_size = config['patch_size']
    print("Loaded trained U-Net model.")

    columns = demo_config['columns']
    overlays = demo_config['overlays']

    num_cols_to_show = get_num_cols_to_show(columns=columns,
                                            overlays=overlays)

    figure_size = tuple(demo_config['figure_size'])
    fig, axs = plt.subplots(len(image_label_pair_file_paths), num_cols_to_show, figsize=figure_size)
    if len(image_label_pair_file_paths) == 1 and num_cols_to_show == 1:
        axs = np.array([axs])
    axs = axs.reshape((-1, num_cols_to_show))

    print()
    for row,image_label_pair_file_path in enumerate(image_label_pair_file_paths):
        image_file_path = Path(image_label_pair_file_path['image'])
        label_file_path = image_label_pair_file_path['label']
        if label_file_path is not None:
            label_file_path = Path(label_file_path)

        print(f"Generating demo #{row+1} (Image: {image_file_path})...")
        plot_demo_row(image_file_path=image_file_path,
                      label_file_path=label_file_path,
                      model=model,
                      device=device,
                      patch_size=patch_size,
                      demo_config=demo_config,
                      axs=axs,
                      row=row)

        print()
    plt.show()
