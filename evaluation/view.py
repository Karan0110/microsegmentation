from pathlib import Path
from pprint import pprint
import os
import numpy as np
from typing import Tuple, Union, List, Optional, Dict, Any
from dotenv import load_dotenv
import argparse
from random import shuffle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu

from segmentation.utils import get_device
from global_utils import load_json5

from segmentation.models.inference import query_inference
from segmentation.utils.serialization import load_model

from global_utils.arguments import get_path_argument
from global_utils import load_grayscale_image

from evaluate import calculate_metrics

def display_informative_segmentation(original_image: np.ndarray, mask : Optional[np.ndarray], segmentation: np.ndarray):
    # Calculate the necessary information
    threshold = threshold_otsu(segmentation.flatten())
    hard_segmentation_otsu = (segmentation > threshold).astype(np.float32)
    
    # Set up figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    # Original image
    axs[0, 0].imshow(original_image, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    
    # Segmentation
    axs[0, 1].imshow(segmentation, cmap='viridis', vmax=0.3, vmin=0.0)
    axs[0, 1].set_title('Segmentation')
    axs[0, 1].axis('off')
    
    # Hard segmentation - Otsu
    axs[1, 0].imshow(hard_segmentation_otsu, cmap='gray')
    axs[1, 0].set_title('Hard Segmentation (Otsu)')
    axs[1, 0].axis('off')
    
    # Ground Truth Mask
    if mask is not None:
        axs[1, 1].imshow(mask, cmap='gray')
        axs[1, 1].set_title('Ground Truth')
    axs[1, 1].axis('off')

    # Adjust layout for plots
    plt.tight_layout()
    
    plt.show()

def parse_args() -> argparse.Namespace:
    # Parse CL arguments
    parser = argparse.ArgumentParser(
        description="Show segmentation of specified image",
    )

    parser.add_argument('-dn', '--dataset', 
                        type=str, 
                        required=True,
                        help='Name of dataset')

    parser.add_argument('-i', '--input', 
                        type=str, 
                        required=True,
                        help='Name of image to segment')

    parser.add_argument('-md', '--modeldir', 
                        type=Path, 
                        help='Models Path (Leave blank to use environment variable value)')

    parser.add_argument('-n', '--name',
                        type=str,
                        required=True,
                        help="Model name")

    parser.add_argument('-ow', '--overwrite', 
                        action='store_true', 
                        help='Overwrite inferences, even when inferences saved to file')
    
    parser.add_argument('-sd', '--savedir',
                        type=str,
                        help="Path to inference save files (for caching)")

    parser.add_argument('-v', '--verbose', 
                        action='store_true', 
                        help='Increase output verbosity')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    dotenv_path = Path(os.environ['PYTHONPATH']) / '.env'
    load_dotenv(dotenv_path=dotenv_path)

    verbose = args.verbose
    overwrite_save_file = args.overwrite

    device = get_device(verbose=verbose)

    models_path = get_path_argument(cl_args=args,
                                    cl_arg_name='modeldir',
                                    env_var_name='MODELS_PATH')

    save_dir = get_path_argument(cl_args=args,
                                    cl_arg_name='savedir',
                                    env_var_name='INFERENCE_SAVE_DIR')

    image_file_path : Path = Path(os.environ['DATA_DIR']) / args.dataset / "Images" / f"{args.input}.png"
    if not image_file_path.is_absolute():
        image_file_path = Path(os.environ['PYTHONPATH']) / image_file_path
    
    mask_file_path = image_file_path.parent.parent / "Masks" / image_file_path.name

    model_name = args.name

    model_dir = models_path / model_name

    if verbose:
        print(f"\nModel directory: {model_dir}")

    config = load_json5(model_dir / 'config.json5')
    model_file_path = model_dir / "model.pth"
    
    patch_size = config['model']['patch_size']

    model = load_model(device=device,
                        config=config,
                        model_dir=model_dir,
                        verbose=verbose)

    segmentation = query_inference(model=model,
                                    device=device,
                                    image_file_path=image_file_path,
                                    model_file_path=model_file_path,
                                    patch_size=patch_size,
                                    save_dir=save_dir,
                                    overwrite_save_file=overwrite_save_file,
                                    verbose=verbose)
    
    original_image = load_grayscale_image(image_file_path)
    mask = load_grayscale_image(mask_file_path) if mask_file_path.exists() else None
    
    # Show metrics in stdout
    if mask is not None:
        metrics_dict = calculate_metrics(segmentation.flatten(), mask.flatten())
        print("\n" + ('#' * 30) + '\n')
        for key, value in metrics_dict.items():
            print(f"{key}: {value:.3f}")
        print("\n" + ('#' * 30) + '\n')

    display_informative_segmentation(original_image, mask, segmentation)
