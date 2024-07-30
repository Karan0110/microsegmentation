from pathlib import Path
import os
import random
from typing import Tuple, Union, List
import json5
import sys

import numpy as np
import matplotlib.pyplot as plt

import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from inference import get_segmentation, get_hard_segmentation
from load import load_json5_config, load_model
from device import get_device

def load_grayscale_image(image_file_path : Path) -> np.ndarray:
    image = np.array(Image.open(image_file_path).convert('L'))
    image = image.astype(np.float32) / 255.

    return image

def get_demo_information(model : nn.Module,
                         device : torch.device,
                         demo_config : dict, 
                         image_file_path : Path,
                         label_file_path : Union[Path, None],
                         patch_size : int,
                         verbose : bool = False,
                         use_caching : bool = False,
                         model_file_path : Union[Path, None] = None) -> Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, np.ndarray]:
    image = load_grayscale_image(image_file_path=image_file_path)
    if label_file_path is not None:
        label = load_grayscale_image(image_file_path=label_file_path)
    else:
        label = None

    segmentation = get_segmentation(image=image,
                                    model=model,
                                    device=device,
                                    patch_size=patch_size,
                                    use_caching=use_caching,
                                    image_file_path=image_file_path,
                                    model_file_path=model_file_path)

    segmentation_threshold : float = demo_config['hard_segmentation_threshold']

    hard_segmentation = get_hard_segmentation(segmentation=segmentation,
                                                segmentation_threshold=segmentation_threshold)

    return (image, label, segmentation, hard_segmentation)

def get_demo_file_paths(demo_config : dict) -> List[Tuple[Path, Union[Path, None]]]:
    raw_demo_file_paths : List[dict] = demo_config['demo_file_paths']

    if demo_config['shuffle_data']:
        random.shuffle(raw_demo_file_paths)
    max_num_demos = demo_config['max_num_demos']
    if max_num_demos is not None:
        raw_demo_file_paths = raw_demo_file_paths[:max_num_demos]

    demo_file_paths : List[Tuple[Path, Union[Path, None]]] = []
    for raw_demo_file_path_pair in raw_demo_file_paths:
        raw_image_file_path = raw_demo_file_path_pair['image']
        raw_label_file_path = raw_demo_file_path_pair['label']

        image_file_path = Path(raw_image_file_path)
        if raw_label_file_path is None:
            label_file_path = None
        else:
            label_file_path = Path(raw_label_file_path)

        demo_file_paths.append((image_file_path, label_file_path))

    return demo_file_paths

def get_colored_image(image : np.ndarray,
                      color : Tuple[float, float, float] = (1., 0., 0.)) -> np.ndarray:
    np_color = np.array(color)

    colored_image = np.empty((*image.shape, 4))
    colored_image[:, :, :3] = np_color
    colored_image[:, :, 3] = image

    return colored_image

def plot_demo(demo_config : dict,
              model : nn.Module,
              device : torch.device,
              patch_size : int,
              model_file_path : Path,
              image_file_path : Path,
              label_file_path : Union[Path, None],
              verbose : bool = True,
              save_file_path : Union[Path, None] = None) -> None:
    if verbose:
        print(f"Showing demo of model: {model_file_path} on image:")
        print(f"{image_file_path}")

    image, label, segmentation, hard_segmentation = get_demo_information(model=model,
                                                                         device=device,
                                                                         demo_config=demo_config,
                                                                         image_file_path=image_file_path,
                                                                         label_file_path=label_file_path,
                                                                         patch_size=patch_size,
                                                                         verbose=False,
                                                                         use_caching=True,
                                                                         model_file_path=model_file_path)

    plot_rows = 2 if (label is None) else 3
    plot_cols = 2
    fig, axs = plt.subplots(plot_rows, plot_cols)#, figsize=(10 * plot_cols, 2 * plot_rows))
    axs = axs.flatten()
    axs_index = 0

    for ax in axs:
        ax.axis('off')

    axs[axs_index].imshow(image, cmap='gray') #type: ignore
    axs[axs_index].axis('off') #type: ignore
    axs[axs_index].set_title("Image") #type: ignore
    axs_index += 1 #type: ignore

    colored_segmentation = get_colored_image(segmentation)
    axs[axs_index].imshow(colored_segmentation) #type: ignore
    axs[axs_index].axis('off') #type: ignore
    axs[axs_index].set_title("Soft Segmentation") #type: ignore
    axs_index += 1 #type: ignore

    axs[axs_index].imshow(hard_segmentation, cmap='gray') #type: ignore
    axs[axs_index].axis('off') #type: ignore
    axs[axs_index].set_title("Hard Segmentation") #type: ignore
    axs_index += 1 #type: ignore

    colored_hard_segmentation = get_colored_image(hard_segmentation)
    axs[axs_index].imshow(colored_hard_segmentation) #type: ignore
    axs[axs_index].imshow(image, cmap='gray', alpha=0.7) #type: ignore
    axs[axs_index].axis('off') #type: ignore
    axs[axs_index].set_title("Hard Segmentation Overlaid") #type: ignore
    axs_index += 1 #type: ignore

    if label is not None:
        axs[axs_index].imshow(label, cmap='gray') #type: ignore
        axs[axs_index].axis('off') #type: ignore
        axs[axs_index].set_title("Ground Truth") #type: ignore
        axs_index += 1 #type: ignore

        colored_label = get_colored_image(label)
        axs[axs_index].imshow(colored_label) #type: ignore
        axs[axs_index].imshow(image, cmap='gray', alpha=0.7) #type: ignore
        axs[axs_index].axis('off') #type: ignore
        axs[axs_index].set_title("Ground Truth Overlaid") #type: ignore
        axs_index += 1 #type: ignore

    if save_file_path is None:
        plt.show()
    else:
        plt.savefig(save_file_path, format='png')
        if verbose:
            print(f"Saved demo to {save_file_path}\n")

if __name__ == '__main__':
    # Validate command line arguments
    if (len(sys.argv)-1) not in [1,]:
        print("Too few / many command-line arguments!")
        print("Correct usage of program:")
        print(f"python3 {sys.argv[0]} [demo_config_path]")
        exit(1)

    #  Load the demo config file
    demo_config_file_path = Path(sys.argv[1])
    demo_config = load_json5_config(demo_config_file_path)

    model_file_path = Path(demo_config['model_file_path'])

    model_config_file_path : Union[Path, None]
    raw_model_config_file_path = demo_config['model_config_file_path']
    if raw_model_config_file_path is None:
        model_config_file_path = None
    else:
        model_config_file_path = Path(raw_model_config_file_path)

    demo_save_dir = Path(demo_config['demo_save_dir']) / model_file_path.stem
    os.makedirs(demo_save_dir, exist_ok=True)

    model, model_config = load_model(model_file_path=model_file_path,
                                     model_config_file_path=model_config_file_path)
    patch_size = model_config['patch_size']
    device = get_device(verbose=True)

    demo_file_paths = get_demo_file_paths(demo_config=demo_config)

    for image_file_path, label_file_path in demo_file_paths:
        demo_save_file_path = demo_save_dir / f"DEMO_{image_file_path.stem}.png"

        plot_demo(demo_config=demo_config,
                model=model,
                device=device,
                patch_size=patch_size,
                model_file_path=model_file_path,
                image_file_path=image_file_path,
                label_file_path=label_file_path,
                verbose=True,
                save_file_path=demo_save_file_path)


# DEMO (v2):
# python3 demo.py demo-config.json5 
