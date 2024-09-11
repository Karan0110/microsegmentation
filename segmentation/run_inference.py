from pathlib import Path
import os
import random
from typing import Tuple, Union, List, Optional
from dotenv import load_dotenv
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from segmentation.utils import get_device
from global_utils import load_json5

from segmentation.models.inference import get_segmentation, get_hard_segmentation
from utils.serialization import load_model

from global_utils.arguments import get_path_argument
from global_utils import load_grayscale_image

from global_utils.hashing import hash_files

def get_segmentation_file_path(output_dir : Path,
                               model_file_path : Path,
                               image_file_path : Path) -> Path:
    file_key = hash_files([model_file_path, image_file_path])
    file_path = output_dir / f"segmentation_{file_key}.npy"
        
    return file_path

def save_inference(model : nn.Module,
                    device : torch.device,
                    image_file_path : Path,
                    model_file_path : Path,
                    patch_size : int,
                    output_dir : Path,
                    update_mode : bool,
                    verbose : bool) -> None:
    image = load_grayscale_image(image_file_path=image_file_path)

    file_path = get_segmentation_file_path(output_dir=output_dir,
                                           model_file_path=model_file_path,
                                           image_file_path=image_file_path)

    if file_path.exists() and update_mode:
        if verbose:
            print(f"Segmentation is already saved to file, skipping...")
    else:
        segmentation = get_segmentation(image=image,
                                        model=model,
                                        device=device,
                                        patch_size=patch_size).to('cpu').numpy()

        if verbose:
            if file_path.exists():
                print(f"Overwriting cache file: {file_path}")
            else:
                print(f"Saving result of inference to cache file: {file_path}")

        os.makedirs(file_path.parent, exist_ok=True)
        np.save(file_path, segmentation)

def parse_args() -> argparse.Namespace:
    # Parse CL arguments
    parser = argparse.ArgumentParser(
        description="Run inference with trained U-Net model."
    )

    # Positional argument (mandatory)
    parser.add_argument('-o', '--output',
                        type=str,
                        help="Save path (Leave blank to use environment variable value)")
    
    parser.add_argument('-i', '--input', 
                        type=Path, 
                        help='Image Data Path (Leave blank to use environment variable value)')
    
    parser.add_argument('-md', '--modeldir', 
                        type=Path, 
                        help='Models Path (Leave blank to use environment variable value)')

    parser.add_argument('-n', '--name',
                        type=str,
                        help="Model name (Leave blank to do all models)")

    parser.add_argument('-u', '--update', 
                        action='store_true', 
                        help='Skip images where inferences already saved to file')

    parser.add_argument('-v', '--verbose', 
                        action='store_true', 
                        help='Increase output verbosity')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    load_dotenv(dotenv_path='../.env')

    verbose = args.verbose
    update_mode = args.update

    device = get_device(verbose=verbose)

    models_path = get_path_argument(cl_args=args,
                                  cl_arg_name='modeldir',
                                  env_var_name='MODELS_PATH')

    if args.name is None:
        model_names = [x.name for x in models_path.iterdir() if (x.is_dir() and not x.name.startswith('.'))]
    else:
        model_names = [args.name]

    for model_name in model_names:
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

        # Output Dir
        output_dir = get_path_argument(cl_args=args,
                                        cl_arg_name='output',
                                        env_var_name='INFERENCE_OUTPUT_PATH')
            
        # Input Dir
        input_dir = get_path_argument(cl_args=args,
                                        cl_arg_name='input',
                                        env_var_name='INFERENCE_INPUT_PATH')


        image_file_paths = list(filter(lambda path: path.suffix == '.png', input_dir.glob("*")))

        for image_file_path in image_file_paths:
            save_inference(model=model,
                            device=device,
                            image_file_path=image_file_path,
                            model_file_path=model_file_path,
                            patch_size=patch_size,
                            output_dir=output_dir,
                            update_mode=update_mode,
                            verbose=verbose)
          