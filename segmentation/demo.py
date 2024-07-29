from pathlib import Path
import random
from typing import Tuple, Union
import json5

import numpy as np

import cv2
from PIL import Image

from inference import get_segmentation, get_hard_segmentation

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

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

def log_demo(writer : SummaryWriter,
             demo_config : dict, 
             model : nn.Module,
             device : str,
             patch_size : int,
             num_epochs : int,
             verbose : bool = False,
             use_caching : bool = False,
             model_file_path : Union[Path, None] = None) -> None:
    # Load demo files
    image_label_pair_file_paths = demo_config['image_label_pair_file_paths']
    if demo_config['shuffle_data']:
        random.shuffle(image_label_pair_file_paths)
    max_num_demos = demo_config['max_num_demos']
    if max_num_demos is not None:
        image_label_pair_file_paths = image_label_pair_file_paths[:max_num_demos]

    if verbose:
        print()
    for demo_index, image_label_pair_file_path in enumerate(image_label_pair_file_paths):
        image_file_path = Path(image_label_pair_file_path['image'])
        label_file_path = image_label_pair_file_path['label']
        if label_file_path is not None:
            label_file_path = Path(label_file_path)

        if verbose: 
            print(f"Generating demo #{demo_index+1} (Image: {image_file_path})...")

        image = load_grayscale_image(image_file_path=image_file_path)
        if label_file_path is not None:
            label = load_grayscale_image(image_file_path=label_file_path)

        segmentation = get_segmentation(image=image,
                                        model=model,
                                        device=device,
                                        patch_size=patch_size,
                                        use_caching=use_caching,
                                        model_file_path=model_file_path)

        segmentation_threshold : float = demo_config['hard_segmentation_threshold']
        hard_segmentation = get_hard_segmentation(segmentation=segmentation,
                                                    segmentation_threshold=segmentation_threshold)


        writer.add_image(f'images/image_{demo_index+1}', image, global_step=num_epochs, dataformats='HW')
        if label_file_path is not None:
            writer.add_image(f'ground_truths/ground_truth_{demo_index+1}', label, global_step=num_epochs, dataformats='HW') #type: ignore

        writer.add_image(f'soft_segmentations/segmentation_{demo_index+1}', segmentation, global_step=num_epochs, dataformats='HW')
        writer.add_image(f'hard_segmentations/segmentation_{demo_index+1}', hard_segmentation, global_step=num_epochs, dataformats='HW')
        
        writer.add_histogram(f'segmentation_histograms/histogram_{demo_index+1}', segmentation, global_step=num_epochs)

    if verbose:
        print()
