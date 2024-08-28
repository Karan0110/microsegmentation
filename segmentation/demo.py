from pathlib import Path
import os
import random
from typing import Tuple, Union, List, Optional
from dotenv import load_dotenv
import argparse

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn

from segmentation.utils import get_device
from global_utils import load_json5

from segmentation.models.inference import get_segmentation, get_hard_segmentation
from utils.checkpoint import load_model

from global_utils.arguments import get_path_argument
from global_utils import load_grayscale_image

def get_demo_information(model : nn.Module,
                         device : torch.device,
                         image_file_path : Path,
                         label_file_path : Union[Path, None],
                         patch_size : int,
                         hard_segmentation_threshold_quantile : float,
                         verbose : bool) -> Tuple[np.ndarray, Union[np.ndarray, None], np.ndarray, np.ndarray]:
    image = load_grayscale_image(image_file_path=image_file_path)
    if label_file_path is not None:
        label = load_grayscale_image(image_file_path=label_file_path)
    else:
        label = None

    segmentation = get_segmentation(image=image,
                                    model=model,
                                    device=device,
                                    patch_size=patch_size).to('cpu').numpy()

    hard_segmentation_threshold = np.quantile(segmentation.flatten(), hard_segmentation_threshold_quantile)
    hard_segmentation = get_hard_segmentation(segmentation=segmentation,
                                              segmentation_threshold=hard_segmentation_threshold) 

    return (image, label, segmentation, hard_segmentation)

def get_demo_file_paths(demo_input_dir : Path) -> List[Tuple[Path, Union[Path, None]]]:
    demo_image_file_paths = filter(lambda path: path.suffix == '.png', (demo_input_dir / "Images").glob("*"))

    demo_file_paths : List[Tuple[Path, Union[Path, None]]] = []
    for demo_image_file_path in demo_image_file_paths:
        demo_mask_file_path = demo_image_file_path.parent.parent / "Masks" / demo_image_file_path.name
        if not demo_mask_file_path.exists():
            demo_mask_file_path = None

        demo_file_paths.append((demo_image_file_path, demo_mask_file_path))

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
              image_file_path : Path,
              label_file_path : Union[Path, None],
              hard_segmentation_threshold_quantile : float,
              save_file_dir : Path,
              verbose : bool = False) -> None:
    image, label, segmentation, hard_segmentation = get_demo_information(model=model,
                                                                         device=device,
                                                                         image_file_path=image_file_path,
                                                                         label_file_path=label_file_path,
                                                                         patch_size=patch_size,
                                                                         hard_segmentation_threshold_quantile=hard_segmentation_threshold_quantile,
                                                                         verbose=verbose)
                                                                        
    plot_rows = 2 if (label is None) else 3
    plot_cols = 2
    fig, axs = plt.subplots(plot_rows, plot_cols)
    axs = axs.flatten()
    axs_index = 0

    for ax in axs:
        ax.axis('off')

    axs[axs_index].imshow(image, cmap='gray') 
    axs[axs_index].axis('off') 
    axs[axs_index].set_title("Image") 
    axs_index += 1 

    colored_segmentation = get_colored_image(segmentation)
    axs[axs_index].imshow(segmentation, cmap='viridis') 
    axs[axs_index].axis('off') 
    axs[axs_index].set_title("Soft Segmentation") 
    axs_index += 1 

    axs[axs_index].imshow(hard_segmentation, cmap='gray') 
    axs[axs_index].axis('off') 
    axs[axs_index].set_title(f"Hard Segmentation\n(Quantile {hard_segmentation_threshold_quantile})")
    axs_index += 1 

    colored_hard_segmentation = get_colored_image(hard_segmentation)
    axs[axs_index].imshow(colored_hard_segmentation) 
    axs[axs_index].imshow(image, cmap='gray', alpha=0.7) 
    axs[axs_index].axis('off') 
    axs[axs_index].set_title("Hard Segmentation Overlaid") 
    axs_index += 1 

    if label is not None:
        axs[axs_index].imshow(label, cmap='gray') 
        axs[axs_index].axis('off') 
        axs[axs_index].set_title("Ground Truth") 
        axs_index += 1 

        colored_label = get_colored_image(label)
        axs[axs_index].imshow(colored_label) 
        axs[axs_index].imshow(image, cmap='gray', alpha=0.7) 
        axs[axs_index].axis('off') 
        axs[axs_index].set_title("Ground Truth Overlaid") 
        axs_index += 1 

    save_file_path = save_file_dir / f"DEMO_{image_file_path.stem}.png"

    plt.savefig(save_file_path, 
                format='png',
                dpi=800)
    if verbose:
        print(f"\nSaved demo to {save_file_path}")

    # Show log scale histogram of segmentation probabilities
    plt.clf()
    plt.cla()

    spectrum = segmentation.flatten()

    plt.hist(spectrum, bins=30)
    plt.yscale('log')

    plt.xlabel('Predicted Pixel Probability')
    plt.ylabel("Frequency Density")

    os.makedirs(save_file_dir / "ProbabilitySpectrums_LogScale", exist_ok=True)
    save_file_path = save_file_dir / "ProbabilitySpectrums_LogScale" / f"{image_file_path.stem}.png"

    dpi = demo_config['save_file_dpi']
    plt.savefig(save_file_path, 
                format='png',
                dpi=dpi)
    if verbose:
        print(f"Saved log-scale probability spectrum to {save_file_path}")

    # Show histogram of segmentation probabilities
    plt.clf()
    plt.cla()

    plt.hist(spectrum, bins=30)
    # plt.yscale('log')

    # hard_segmentation_threshold = np.quantile(spectrum, 0.9)
    # right_spectrum = spectrum[spectrum >= hard_segmentation_threshold]
    # plt.hist(right_spectrum, bins=30)

    plt.xlabel('Predicted Pixel Probability')
    plt.ylabel("Frequency Density")

    os.makedirs(save_file_dir / "ProbabilitySpectrums", exist_ok=True)
    save_file_path = save_file_dir / "ProbabilitySpectrums" / f"{image_file_path.stem}.png"

    dpi = demo_config['save_file_dpi']
    plt.savefig(save_file_path, 
                format='png',
                dpi=dpi)
    if verbose:
        print(f"Saved probability spectrum to {save_file_path}")

def plot_demos(demo_config : dict,
               hard_segmentation_threshold_quantile: float,
               model_dir : Path,
               demo_save_dir : Path,
               demo_input_dir : Path,
               model : nn.Module,
               model_name : str,
               model_config : dict,
               device : torch.device,
               demo_name : str,
               verbose : bool = False) -> None:
    os.makedirs(demo_save_dir, exist_ok=True)

    patch_size = model_config['patch_size']

    demo_file_paths = get_demo_file_paths(demo_input_dir=demo_input_dir)

    for image_file_path, label_file_path in demo_file_paths:
        plot_demo(demo_config=demo_config,
                model=model,
                device=device,
                patch_size=patch_size,
                image_file_path=image_file_path,
                hard_segmentation_threshold_quantile=hard_segmentation_threshold_quantile,
                label_file_path=label_file_path,
                save_file_dir=demo_save_dir,
                verbose=verbose)
    
    plt.close()

def parse_args() -> argparse.Namespace:
    # Parse CL arguments
    parser = argparse.ArgumentParser(
        description="Demonstrate trained U-Net model."
    )

    # Positional argument (mandatory)
    parser.add_argument('-c', '--config', 
                        type=Path, 
                        help='Config File Path (Leave blank to use environment variable value)')
    
    parser.add_argument('-sd', '--savedir', 
                        type=Path, 
                        help='Demo Save Path (Leave blank to use environment variable value)')
    
    parser.add_argument('-dd', '--datadir', 
                        type=Path, 
                        help='Demo Data Path (Leave blank to use environment variable value)')
    
    parser.add_argument('-i', '--inputdir', 
                        type=Path, 
                        help='Demo Input Path (Leave blank to use environment variable value)')
    
    parser.add_argument('-md', '--modeldir', 
                        type=Path, 
                        help='Model Path (Leave blank to use environment variable value)')

    parser.add_argument('-n', '--name',
                        type=str,
                        required=True,
                        help="Model name")

    parser.add_argument('-t', '--threshold',
                        type=float,
                        help="Threshold quantile for hard segmentation (Leave blank to use value in config file)")

    parser.add_argument('-dn', '--demoname',
                        type=str,
                        default=None,
                        help="Name of demo save file (Leave blank to use same name as model)")

    parser.add_argument('-v', '--verbose', 
                        action='store_true', 
                        help='Increase output verbosity')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    load_dotenv()

    verbose = args.verbose

    # Load model
    model_name = args.name

    model_dir = get_path_argument(cl_args=args,
                                  cl_arg_name='modeldir',
                                  env_var_name='MODELS_PATH')
    model_dir = model_dir / model_name

    device = get_device(verbose=verbose)

    config = load_json5(model_dir / 'config.json5')

    model = load_model(model_config=config['model'],
                       device=device,
                       save_dir=model_dir,
                       verbose=verbose)

    patch_size = config['model']['patch_size']

    #  Load the demo config file
    demo_config_file_path = get_path_argument(cl_args=args,
                                              cl_arg_name='config',
                                              env_var_name='DEMO_CONFIG_PATH') 
    demo_config = load_json5(demo_config_file_path)
    if not isinstance(demo_config, dict):
        raise ValueError(f"Invalid demo config! Must be a dict")

    # Demo Name
    if args.demoname is not None:
        demo_name = args.demoname
    else:
        demo_name = model_name

    # Demo Save Dir
    demo_save_dir = get_path_argument(cl_args=args,
                                      cl_arg_name='savedir',
                                      env_var_name='DEMO_SAVE_PATH')
    demo_save_dir = demo_save_dir / demo_name
        
    # Demo Input Dir
    demo_input_dir = get_path_argument(cl_args=args,
                                      cl_arg_name='inputdir',
                                      env_var_name='DEMO_INPUT_PATH')
    
    # Hard segmentation threshold
    hard_segmentation_threshold_quantile = args.threshold
    if hard_segmentation_threshold_quantile is None:
        hard_segmentation_threshold_quantile = float(demo_config['hard_segmentation_threshold_quantile'])

    if verbose:
        print(f"\nModel directory: {model_dir}")

    plot_demos(demo_config=demo_config,
               model_name=model_name,
               hard_segmentation_threshold_quantile=hard_segmentation_threshold_quantile,
               model=model,
               model_dir=model_dir,
               demo_save_dir=demo_save_dir,
               demo_input_dir=demo_input_dir,
               demo_name=demo_name,
               model_config=config['model'],
               device=device,
               verbose=verbose)
