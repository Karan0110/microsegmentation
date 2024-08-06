from pathlib import Path
import os
import random
from typing import Tuple, Union, List, Optional
import json5
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

import cv2
from PIL import Image

import torch
import torch.nn as nn

from models.inference import get_segmentation, get_hard_segmentation
from utils import load_json5, get_device
from models.serialization import load_model_from_file, get_model_save_file_paths

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
                         verbose : bool,
                         hard_segmentation_threshold : float,
                         model_file_path : Union[Path, None] = None) -> Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, np.ndarray]:
    image = load_grayscale_image(image_file_path=image_file_path)
    if label_file_path is not None:
        label = load_grayscale_image(image_file_path=label_file_path)
    else:
        label = None

    segmentation = get_segmentation(image=image,
                                    model=model,
                                    device=device,
                                    patch_size=patch_size).to('cpu').numpy()

    hard_segmentation = get_hard_segmentation(segmentation=segmentation,
                                              segmentation_threshold=hard_segmentation_threshold) #type: ignore

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
              hard_segmentation_threshold : float,
              save_to_file : bool,
              verbose : bool = True,
              only_show_histogram : bool = False,
              save_file_dir : Union[Path, None] = None) -> None:
    if verbose:
        print(f"\nShowing demo of model: {model_file_path} on image:")
        print(f"{image_file_path}")

    image, label, segmentation, hard_segmentation = get_demo_information(model=model,
                                                                         device=device,
                                                                         demo_config=demo_config,
                                                                         image_file_path=image_file_path,
                                                                         label_file_path=label_file_path,
                                                                         patch_size=patch_size,
                                                                         hard_segmentation_threshold=hard_segmentation_threshold,
                                                                         verbose=False,
                                                                         model_file_path=model_file_path)
                                                                        
    if not only_show_histogram:                                        
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
        axs[axs_index].set_title(f"Hard Segmentation\n(Threshold {hard_segmentation_threshold})") #type: ignore
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

        if save_to_file:
            if save_file_dir is None:
                raise ValueError(f"No save file dir provided!")

            save_file_path = save_file_dir / f"DEMO_{image_file_path.stem}.png"
    
            plt.savefig(save_file_path, 
                        format='png',
                        dpi=800)
            if verbose:
                print(f"\nSaved demo to {save_file_path}")
        else:
            plt.show()

    # Show histogram of segmentation probabilities
    plt.clf()
    plt.cla()

    plt.hist(segmentation.flatten(), bins=30)

    if save_to_file:
        if save_file_dir is None:
            raise ValueError(f"No save file dir provided!")

        save_file_path = save_file_dir / f"PROB_HIST_{image_file_path.stem}.png"
 
        dpi = demo_config['save_file_dpi']
        plt.savefig(save_file_path, 
                    format='png',
                    dpi=dpi)
        if verbose:
            print(f"Saved probability histogram to {save_file_path}")
    else:
        plt.show()

def plot_demos(demo_config : dict,
               hard_segmentation_threshold : float,
               model_dir : Path,
               model : nn.Module,
               model_name : str,
               model_config : dict,
               device : torch.device,
               verbose : bool,
               save_to_file : bool,
               only_show_histogram : bool = False,
               demo_name : Optional[str] = None) -> None:
    if demo_name is None:
        demo_name = model_name

    state_save_file_path, config_save_file_path = get_model_save_file_paths(model_dir=model_dir,
                                                                            model_name=model_name)

    demo_save_dir = Path(demo_config['demo_save_dir']) / demo_name
    os.makedirs(demo_save_dir, exist_ok=True)

    patch_size = model_config['patch_size']

    demo_file_paths = get_demo_file_paths(demo_config=demo_config)

    for image_file_path, label_file_path in demo_file_paths:
        plot_demo(demo_config=demo_config,
                model=model,
                device=device,
                patch_size=patch_size,
                model_file_path=state_save_file_path,
                image_file_path=image_file_path,
                hard_segmentation_threshold=hard_segmentation_threshold,
                label_file_path=label_file_path,
                save_file_dir=demo_save_dir,
                only_show_histogram=only_show_histogram,
                verbose=verbose,
                save_to_file=save_to_file)


if __name__ == '__main__':
    # Parse CL arguments
    parser = argparse.ArgumentParser(
        description="Demonstrate trained U-Net model."
    )

    # Positional argument (mandatory)
    parser.add_argument('-c', '--config', 
                        type=Path, 
                        required=True,
                        help='Config File Path')

    parser.add_argument('-n', '--name',
                        type=str,
                        required=True,
                        help="Model name")

    parser.add_argument('-t', '--threshold',
                        type=float,
                        help="Threshold for hard segmentation (Else use default value in config file)")

    parser.add_argument('-v', '--verbose', 
                        action='store_true', 
                        help='Increase output verbosity')

    parser.add_argument('-dn', '--demoname',
                        type=str,
                        default=None,
                        help="Name of demo save file (Leave blank to use same name as model)")

    parser.add_argument('--show', 
                        action='store_true', 
                        help='Show demos in window instead of saving to files')

    args = parser.parse_args()

    verbose = args.verbose

    show_mode = args.show

    #  Load the demo config file
    demo_config_file_path = args.config
    demo_config = load_json5(demo_config_file_path)
    if not isinstance(demo_config, dict):
        raise ValueError(f"Invalid demo config! Must be a dict")

    # hard segmentation threshold
    hard_segmentation_threshold = args.threshold
    if hard_segmentation_threshold is None:
        hard_segmentation_threshold = float(demo_config['hard_segmentation_threshold'])

    # Load model
    model_name = args.name
    model_dir = Path(demo_config['model_dir'])

    state_save_file_path, config_save_file_path = get_model_save_file_paths(model_dir=model_dir,
                                                                            model_name=model_name)

    model, config = load_model_from_file(model_file_path=state_save_file_path,
                                     config_file_path=config_save_file_path)
                        
    patch_size = config['model']['patch_size']
    device = get_device(verbose=verbose)
        
    plot_demos(demo_config=demo_config,
               model_name=model_name,
               hard_segmentation_threshold=hard_segmentation_threshold,
               model=model,
               model_dir=model_dir,
               model_config=config['model'],
               device=device,
               verbose=verbose,
               save_to_file=(not show_mode))
