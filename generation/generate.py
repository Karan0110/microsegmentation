import os
import itertools
import sys
import json5
import time
import random
from pathlib import Path
from typing import Iterable, Tuple, Union, List, Optional
import argparse
import re
from dotenv import load_dotenv

import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt

from PIL import Image

from fluorophore_simulation import get_mt_points, get_fluorophore_points, get_fluorophore_image, calculate_fluorophore_emission_per_second
from microscope_simulation import get_psf_kernel, get_digital_signal, quantize_intensities
from depoly_simulation import simulate_depoly

from visualize import visualize
from labelling import get_mask

from global_utils.load_json5 import load_json5
from global_utils.parse_number_or_range import parse_number_or_range

def pad_and_tile_sample(arr, tile_size):
    """
    Pads the array to make its dimensions a multiple of tile_size, then breaks it
    into tiles of size tile_size x tile_size and samples a value uniformly at random
    from each tile.
    
    Args:
    - arr: NumPy array of shape (H, W)
    - tile_size: Size of the tiles (default is 10)
    
    Returns:
    - A new array where each element is a randomly sampled value from the corresponding tile.
    """
    # Get original dimensions
    H, W = arr.shape

    # Calculate padding needed to make H and W multiples of tile_size
    pad_h = (tile_size - (H % tile_size)) % tile_size
    pad_w = (tile_size - (W % tile_size)) % tile_size

    # Pad the array with zeros
    arr_padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    # Get the new padded dimensions
    H_padded, W_padded = arr_padded.shape

    # Reshape the array to get tiles of tile_size x tile_size
    arr_reshaped = arr_padded.reshape(H_padded // tile_size, tile_size, W_padded // tile_size, tile_size)

    # Initialize the output array to store the sampled values
    result = np.zeros((H_padded // tile_size, W_padded // tile_size))

    # Sample uniformly from each tile
    for i in range(H_padded // tile_size):
        for j in range(W_padded // tile_size):
            tile = arr_reshaped[i, :, j, :]  # Extract the tile (10x10 sub-array)
            tile_flat = tile.flatten()  # Flatten the tile to a 1D array
            result[i, j] = np.random.choice(tile_flat)  # Randomly sample one value from the tile

    return result

def pad_and_tile_aggregate(arr, 
                           tile_size, 
                           agg_func):
    """
    Pads the array to make its dimensions a multiple of tile_size, then breaks it
    into tiles of size tile_size x tile_size and applies the aggregation function agg_func.
    
    Args:
    - arr: NumPy array of shape (H, W)
    - tile_size: Size of the tiles (default is 10)
    - agg_func: Aggregation function (e.g., np.max, np.mean)
    
    Returns:
    - A new array where each element is the result of applying agg_func to the corresponding tile.
    """
    # Get original dimensions
    H, W = arr.shape

    # Calculate padding needed to make H and W multiples of tile_size
    pad_h = (tile_size - (H % tile_size)) % tile_size
    pad_w = (tile_size - (W % tile_size)) % tile_size

    # Pad the array with zeros
    arr_padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    # Get the new padded dimensions
    H_padded, W_padded = arr_padded.shape

    # Reshape the array to get tiles of tile_size x tile_size
    arr_reshaped = arr_padded.reshape(H_padded // tile_size, tile_size, W_padded // tile_size, tile_size)

    # Apply the aggregation function (agg_func) on each tile
    arr_aggregated = agg_func(arr_reshaped, axis=(1, 3))

    return arr_aggregated

def generate(tubulaton_output_file_path : Path,
             config : dict,
             depoly_proportion : float,
             mode : str,
             show_3d : bool,
             verbose : bool) -> Tuple[np.ndarray, np.ndarray]:
    if verbose:
        print(f"Loading MT points from {tubulaton_output_file_path} to numpy array...")
    mt_points, mt_ids = get_mt_points(file_path=tubulaton_output_file_path,
                                      config=config,
                                      verbose=verbose)

    #TODO - It shouldn't really cause issues if no MTs, even though this shouldn't happen if everything went right (But this is a temporary fix)
    if mt_points.shape[0] == 0:
        image = np.zeros((512,512))
        mask = np.zeros((512,512))
        return image, mask
    #TODO END

    unique_mt_ids = np.unique(mt_ids)

    if show_3d and mode == 'demo_interactive':
        print("Showing visualization of tubulin (before depoly simulation)")
        visualize(mt_points)
    
    #TODO - (mt, tubulin)
    tubulin_points, mt_points = simulate_depoly(mt_points=mt_points,
                                                mt_ids=mt_ids,
                                                config=config,
                                                unique_mt_ids=unique_mt_ids,
                                                proportion=depoly_proportion)
    if show_3d and mode == 'demo_interactive':
        print("Showing visualization of tubulin (after depoly simulation)")
        visualize(tubulin_points)

    fluorophore_points = get_fluorophore_points(tubulin_points=tubulin_points,
                                                config=config)

    if show_3d and mode == 'demo_interactive':
        print("Visualizing fluorophore points...")
        visualize(fluorophore_points,
                  colors='yellow',
                  background_color='black')

    if mode.startswith('demo'):
        plot_rows = 2
        plot_cols = 2
        fig, axs = plt.subplots(plot_rows, plot_cols, figsize=(10 * plot_cols, 2 * plot_rows))
        axs = axs.flatten()
        axs_index = 0
    
    #TODO - It shouldn't be the case no FPs - there are a fixed amount! (But this is a temporary fix)
    if fluorophore_points.shape[0] == 0:
        image = np.zeros((512,512))
        mask = np.zeros((512,512))
        return image, mask
    #TODO END
    
    x_min = fluorophore_points[:, 0].min()
    y_min = fluorophore_points[:, 1].min()
    z_min = fluorophore_points[:, 2].min()

    x_max = fluorophore_points[:, 0].max()
    y_max = fluorophore_points[:, 1].max()
    z_max = fluorophore_points[:, 2].max()

    z_slice_position = config['microscope']['z_slice_position']
    z_slice=z_min*z_slice_position + z_max*(1-z_slice_position)

    mask = get_mask(mt_points=mt_points,
                    z_slice=z_slice,
                    config=config,
                    bounding_box=(x_min, y_min, x_max, y_max),
                    verbose=verbose)
        
    fluorophore_image = get_fluorophore_image(fluorophore_points=fluorophore_points,
                                              config=config,
                                              bounding_box=(x_min, y_min, x_max, y_max),
                                              z_slice=z_slice,
                                              verbose=verbose)

    # Crop the image/mask

    x_frame_size = config['x_frame_size']
    y_frame_size = config['y_frame_size']
    H, W = fluorophore_image.shape
    assert (H,W) == mask.shape

    new_max_i = int(y_frame_size * H)
    new_min_i = int(1. - y_frame_size * H)
    new_max_j = int(x_frame_size * W)
    new_min_j = int(1. - x_frame_size * W)

    fluorophore_image = fluorophore_image[new_min_i:new_max_i, new_min_j:new_max_j]
    mask = mask[new_min_i:new_max_i, new_min_j:new_max_j]
    
    if mode.startswith('demo'):
        axs[axs_index].imshow(fluorophore_image, cmap='gray') #type: ignore
        axs[axs_index].axis('off') #type: ignore
        axs[axs_index].set_title("Fluorophore image") #type: ignore
        axs_index += 1 #type: ignore

    fluorophore_emission_per_second  = calculate_fluorophore_emission_per_second(config=config)
    intensities = fluorophore_image * fluorophore_emission_per_second 

    # TODO At present - we can take this (inexpensive) calculation of the kernel out of the loop
    # But eventually want to make the parameters stochastic so the kernel will be recalculated each time
    psf_kernel = get_psf_kernel(config=config)
    image = scipy.signal.convolve2d(intensities, psf_kernel, mode='same')

    image = get_digital_signal(intensities=image,
                               config=config)

    # if verbose:
    #     print("Quantizing intensities...")
    # image = quantize_intensities(image=image,
    #                              config=config)

    # Reduce resolution
    tile_size = config['tile_size']
    if tile_size > 1:
        image = pad_and_tile_sample(image, tile_size=tile_size)
        mask = pad_and_tile_aggregate(mask, tile_size=tile_size, agg_func=np.max)

    # Apply speckly noise
    speckle_noise_mean = config['speckle_noise_mean']
    speckle_noise_std = config['speckle_noise_std']
    speckle_noise = np.maximum(np.random.normal(loc=speckle_noise_mean, scale=speckle_noise_std, size=image.shape), 0.)

    image += speckle_noise * image.max()

    if mode.startswith('demo'):
        axs[axs_index].imshow(image, cmap='gray') #type: ignore
        axs[axs_index].axis('off') #type: ignore
        axs[axs_index].set_title("Synthetic Image") #type: ignore
        axs_index += 1 #type: ignore

    if mode.startswith('demo'):
        axs[axs_index].imshow(mask, cmap='gray') #type: ignore
        axs[axs_index].axis('off') #type: ignore
        axs[axs_index].set_title("Mask Segmentation") #type: ignore
        axs_index += 1 #type: ignore

    if mode.startswith('demo'):
        colored_mask = np.tile(mask[:, :, np.newaxis], (1, 1, 4)).astype(np.float32)
        colored_mask[:, :, 0] = 1.
        colored_mask[:, :, 1] = 0.
        colored_mask[:, :, 2] = 0.

        axs[axs_index].imshow(colored_mask) #type: ignore
        axs[axs_index].imshow(image, cmap='gray', alpha=0.5) #type: ignore
        axs[axs_index].axis('off') #type: ignore
        axs[axs_index].set_title("Mask Segmentation Overlaid") #type: ignore
        axs_index += 1 #type: ignore

    if mode.startswith('demo'):
        plt.tight_layout()

    if verbose:
        print()

    return image, mask

def is_file_name_valid_format(file_path : Path) -> bool:
    stem = file_path.stem

    pattern = r"^tubulaton-\d+$"

    return bool(re.match(pattern, stem))

def extract_file_id(file_path : Path) -> int:
    stem = file_path.stem

    pattern = r"^tubulaton-(\d+)$"
    match = re.match(pattern, stem)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"The file name {file_path} does not match the 'tubulaton-[number].vtk' format.")

def parse_args() -> argparse.Namespace:
    # Parse CL arguments
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data from tubulaton .vtk files."
    )

    parser.add_argument('-n', '--name', 
                        type=str, 
                        required=True,
                        help='Name of Dataset')

    parser.add_argument('-ids', '--ids',
                        type=str,
                        help="Which file ids to use? (Either a number (e.g. 32) or a range (e.g. 32-48)")

    parser.add_argument('-c', '--config',
                        type=str,
                        required=True,
                        help='Name of config file')
    
    parser.add_argument("--depoly",
                        type=float,
                        help="The rate of depolymerization (Leave blank to draw a random value from the distribution specified in config)")

    parser.add_argument('--mode', choices=['demo_interactive', 'demo_headless', 'overwrite', 'update'], 
                        default='update',
                        help="""Mode of operation: demo, overwrite, or update (default: update):
                                - demo_interactive: Interactive demonstration of program on a single .vtk file. No file writing.
                                - demo_headless:    Saves (partial) output of demos to files.
                                - overwrite:        Overwrite any existing synthetic data in specified location
                                - update:           Skip over any pre-existing synthetic data files""")

    train_val_group = parser.add_mutually_exclusive_group()
    train_val_group.add_argument('--train',
                        action='store_true',
                        help="Make all the data training")
    train_val_group.add_argument('--val',
                        action='store_true',
                        help="Make all the data validation")

    parser.add_argument('--show_3d', 
                        action='store_true', 
                        help='Show 3D models')

    parser.add_argument('-v', '--verbose', 
                        action='store_true', 
                        help='Increase output verbosity')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    start_time = time.time()
    random.seed(1000)

    args = parse_args()
    load_dotenv(dotenv_path='../.env')

    mode = args.mode
    show_3d = args.show_3d
    verbose = args.verbose

    dataset_name = args.name

    output_dir = Path(os.environ["GENERATION_OUTPUT_DIR"]) / dataset_name

    tubulaton_output_path = Path(os.environ["TUBULATON_OUTPUT_DIR"]) / dataset_name

    config_file_dir = Path(os.environ["GENERATION_CONFIGS_DIR"])
    config_file_stem = args.config
    config_file_path = config_file_dir / f"{config_file_stem}.json5"

    file_ids : Optional[range]
    file_ids = parse_number_or_range(args.ids) if (args.ids is not None) else args.ids

    config = load_json5(config_file_path)
    if verbose:
        print(f"Loaded config file: {config_file_path}")


    # Get the tubulaton output file paths
    tubulaton_output_file_paths : list
    if file_ids is not None:
        if tubulaton_output_path.is_file():
            raise ValueError(f"Provided file ID range {file_ids}, but didn't give a directory for tubulaton_output_path!")

        tubulaton_output_file_paths = []

        for file_id in file_ids:
            target_tubulaton_output_file_path = tubulaton_output_path / f"tubulaton-{file_id}.vtk" 
            if not target_tubulaton_output_file_path.exists():
                raise FileNotFoundError(f"Provided file ID {file_id} does not correspond to a file:" 
                                      + f"\n{target_tubulaton_output_file_path} is not a file!")
            tubulaton_output_file_paths.append(target_tubulaton_output_file_path)
    else:
        if tubulaton_output_path.is_dir():
            tubulaton_output_file_paths = [file_path for file_path in tubulaton_output_path.iterdir() if file_path.suffix == '.vtk'] 
            tubulaton_output_file_paths = sorted(tubulaton_output_file_paths, key=extract_file_id)
        elif tubulaton_output_path.is_file():
            tubulaton_output_file_paths = [tubulaton_output_path]
        else:
            raise ValueError(f"Provided [tubulaton_output_path]: {tubulaton_output_path} is neither a file nor a directory!")
    
    # Check files have correct name format
    for tubulaton_output_file_path in tubulaton_output_file_paths:
        if not is_file_name_valid_format(tubulaton_output_file_path):
            raise ValueError(f"The file name {tubulaton_output_file_path} does not match the 'tubulaton-[number].vtk' format.")

    # Make iterator for tubulaton output file paths
    file_path_iterator : List[Path]   
    if mode == 'demo_interactive':
        file_path = random.choice(tubulaton_output_file_paths)

        print(f"Demo file choice: {file_path}")
        file_path_iterator = [file_path]
    else:
        file_path_iterator = tubulaton_output_file_paths
        num_uses = config['num_uses']
        file_path_iterator *= num_uses

    # print(f"WARNING: Truncating the file path lists for testing. Must delete this part!!!")
    # file_path_iterator = file_path_iterator[:10]

    train_eval_split = config['train_eval_split']
    train_counter = 1
    eval_counter = 1
    if args.val:
        cutoff = 0
    elif args.train:
        cutoff = len(file_path_iterator)
    else:
        cutoff = int(len(file_path_iterator) * train_eval_split)

    for index, tubulaton_output_file_path in enumerate(file_path_iterator):
        block_start_time = time.time()

        if args.depoly is None:
            depoly_proportion_distribution_info = config['depoly_proportion_distribution']
            depoly_proportion_distribution  = getattr(np.random, depoly_proportion_distribution_info['name'])
            params = depoly_proportion_distribution_info.get('params', {})
            depoly_proportion = depoly_proportion_distribution(**params)
        else:
            depoly_proportion = args.depoly
        if verbose:
            print(f"Using depolymerization rate: {depoly_proportion}")

        file_id = extract_file_id(tubulaton_output_file_path)

        if not mode.startswith('demo'):
            if output_dir is None:
                raise ValueError("Output dir cannot be None when not in demo mode!")

            if index < cutoff:
                output_file_id = train_counter
                train_counter += 1
                subfolder_name = "Train"
            else:
                output_file_id = eval_counter
                eval_counter += 1
                subfolder_name = "Eval"

            image_file_name = f"image-{output_file_id}.png"

            image_file_path = output_dir / subfolder_name / 'Images' / image_file_name
            mask_file_path = output_dir / subfolder_name / 'Masks' / image_file_name

            os.makedirs(image_file_path.parent, exist_ok=True)
            os.makedirs(mask_file_path.parent, exist_ok=True)

            if mode == 'overwrite' or (not image_file_path.exists()) or (not mask_file_path.exists()):
                image, mask = generate(tubulaton_output_file_path=tubulaton_output_file_path,
                                        depoly_proportion=depoly_proportion,
                                        config=config,
                                        mode=mode,
                                        show_3d=show_3d,
                                        verbose=verbose)

                image = image / image.max() * 255.
                image = Image.fromarray(image.astype(np.uint8), mode='L')

                mask = mask * 255
                mask = Image.fromarray(mask, mode='L')

                image.save(image_file_path)
                mask.save(mask_file_path)

                if verbose:
                    print(f"Saved data to {image_file_path} and {mask_file_path}.")
            else:
                if verbose:
                    print(f"Image and mask files for {tubulaton_output_file_path} already exist:")
                    print(f"Image path: {image_file_path}")
                    print(f"mask path: {mask_file_path}")
                    print(f"Skipping this tubulaton file.")
                    print()
        else:
            image, mask = generate(tubulaton_output_file_path=tubulaton_output_file_path,
                                    depoly_proportion=depoly_proportion,
                                    config=config,
                                    mode=mode,
                                    show_3d=show_3d,
                                    verbose=verbose)
            
            if mode == 'demo_interactive':
                plt.show()
            elif mode == 'demo_headless':
                if output_dir is None:
                    output_dir = Path("./demos/")

                os.makedirs(output_dir, exist_ok=True)

                demo_save_file_path = output_dir / f"DEMO_{tubulaton_output_file_path.stem}.png"

                plt.savefig(demo_save_file_path,
                            format='png',
                            dpi=800)

                if verbose:
                    print(f"Saved demo to {demo_save_file_path}")
            else:
                raise ValueError(f"Invalid mode provided: {mode}")
        
        if verbose and mode != 'demo_interactive':
            time_taken = time.time() - block_start_time
            print(f"Took {time_taken:.2f} seconds.")
            print("\n" * 2 + "#" * 130 + "\n"*2)


    time_taken = time.time() - start_time
    if verbose and mode != 'demo_interactive':
        print(f"\n\nTook {time_taken:.2f} seconds in total.")
        if len(tubulaton_output_file_paths) > 1:
            print(f"({time_taken / len(tubulaton_output_file_paths):.2f} seconds per .vtk file)")
