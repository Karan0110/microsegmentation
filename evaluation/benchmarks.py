from pathlib import Path
import os
import numpy as np
from typing import Tuple, Union, List, Optional, Dict, Any
from dotenv import load_dotenv
import argparse
from random import shuffle

import numpy as np
import pandas as pd

from global_utils import load_grayscale_image
from global_utils.iterate_dataset import iterate_image_mask_pairs

from evaluate import update_metrics_df

def parse_args() -> argparse.Namespace:
    # Parse CL arguments
    parser = argparse.ArgumentParser(
        description="Evaluate classical techniques for benchmarking."
    )
    
    parser.add_argument('-i', '--input', 
                        type=str, 
                        required=True,
                        help='Dataset Name')
    
    parser.add_argument('-c', '--count',
                        type=int,
                        default=0,
                        help="Max number of samples to use (Leave blank to use everything in directory)")

    parser.add_argument('-v', '--verbose', 
                        action='store_true', 
                        help='Increase output verbosity')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    load_dotenv(dotenv_path='../.env')

    verbose = args.verbose
    max_num_samples = args.count

    data_dir = Path(os.environ['DATA_DIR']) / args.input
    if not data_dir.is_absolute():
        data_dir = Path(os.environ['PYTHONPATH']) / data_dir

    print(f"Data Dir: {data_dir}")
    image_mask_pairs = sorted(list(iterate_image_mask_pairs(data_dir)), key=lambda pair: pair[0].stem)
    num_samples = len(image_mask_pairs)

    print("Calculating metrics...")

    metrics_df_file_path = Path(os.environ['PYTHONPATH']) / 'evaluation' / "metrics.csv"

    if metrics_df_file_path.exists():
        metrics_df = pd.read_csv(metrics_df_file_path)
    else:
        metrics_df = pd.DataFrame()

    model_name = "brightness_segmentation"

    if verbose:
        print(f"Data Dir: {data_dir}")
    image_mask_pairs = sorted(list(iterate_image_mask_pairs(data_dir)), key=lambda pair: pair[0].stem)

    counter = 0
    for image_file_path, mask_file_path in image_mask_pairs:
        if verbose:
            print(f"Calculating brightness segmentation benchmark for {image_file_path.name}")

        segmentation = load_grayscale_image(image_file_path)
        mask = load_grayscale_image(mask_file_path)

        metrics_df = update_metrics_df(metrics_df,
                                        image_name=image_file_path.stem,
                                        mask=mask,
                                        segmentation=segmentation,
                                        model_name=model_name,
                                        dataset_name=args.input,
                                        verbose=verbose)

        counter += 1
        if max_num_samples is not None and counter > max_num_samples:
            break

    if verbose:
        print(f"Writing metrics to metrics.csv file...")
    metrics_df.to_csv("metrics.csv", index=False)

