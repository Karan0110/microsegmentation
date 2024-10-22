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
import tifffile

from skimage.filters import threshold_otsu

from segmentation.utils import get_device
from global_utils import load_json5

from segmentation.models.inference import query_inference
from segmentation.utils.serialization import load_model

from global_utils.arguments import get_path_argument
from global_utils.parse_number_or_range import parse_number_or_range
from global_utils import load_grayscale_image

from evaluate import calculate_metrics

def parse_args() -> argparse.Namespace:
    # Parse CL arguments
    parser = argparse.ArgumentParser(
        description="Show segmentation of specified image",
    )

    parser.add_argument('-i', '--input', 
                        type=Path, 
                        required=True,
                        help='Path to lsm file')

    parser.add_argument('-z', '--zstack',
                        type=int,
                        required=True,
                        help="Z slice position")

    parser.add_argument('-t', '--times',
                        type=str,
                        help="Time steps to show")

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
    load_dotenv(dotenv_path='../.env')

    verbose = args.verbose
    overwrite_save_file = args.overwrite

    device = get_device(verbose=verbose)

    models_path = get_path_argument(cl_args=args,
                                    cl_arg_name='modeldir',
                                    env_var_name='MODELS_PATH')

    save_dir = get_path_argument(cl_args=args,
                                 cl_arg_name='savedir',
                                 env_var_name='INFERENCE_SAVE_DIR')

    lsm_file_path : Path = args.input
    with tifffile.TiffFile(lsm_file_path) as tif:
        lsm_data = tif.asarray()
    
    metadata = tif.lsm_metadata
    if metadata is None:
        raise ValueError()
    num_time_steps = metadata['DimensionTime']
    
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

    z_stack = args.zstack
    image_file_names = [f"frame_{time_step}_{z_stack}.png" for time_step in range(num_time_steps)]

    assert len(image_file_names) == num_time_steps

    raw_time_steps = args.times
    time_steps = parse_number_or_range(raw_time_steps) if raw_time_steps is not None else range(num_time_steps)
    assert 0 <= min(time_steps) <= max(time_steps) <= num_time_steps

    fig, axs = plt.subplots(len(time_steps), 2, figsize=(10, 2 * num_time_steps))
    print(axs.shape)

    for time_step in time_steps:
        image_file_name = image_file_names[time_step]
        image_file_path = lsm_file_path.parent / f'PNG_{lsm_file_path.stem}' / image_file_name

        segmentation = query_inference(model=model,
                                    device=device,
                                    image_file_path=image_file_path,
                                    model_file_path=model_file_path,
                                    patch_size=patch_size,
                                    save_dir=save_dir,
                                    overwrite_save_file=overwrite_save_file,
                                    verbose=verbose)
        
        original_image = load_grayscale_image(image_file_path)

        ax_index = time_step - min(time_steps)

        axs[ax_index, 0].imshow(original_image, cmap='viridis', vmax=0.5)#,vmin=0.0, vmax=1.0)
        axs[ax_index, 0].axis('off')

        axs[ax_index, 1].imshow(segmentation, cmap='viridis', vmax=0.5)#,vmin=0.0, vmax=1.0)
        axs[ax_index, 1].axis('off')

    plt.show()
        
