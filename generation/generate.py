# Command line arguments
# output_dir                 : Location to save data
# sample_name                : A unique ID for the data sample
# tubulaton_output_file_path : Path of tubulaton .vtk output file

import os
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
from labelling import get_label

from visualize import visualize
from global_utils.load_json5 import load_json5

def generate(tubulaton_output_file_path : Path,
             config : dict,
             depoly_proportion : float,
             mode : str,
             verbose : bool) -> Tuple[np.ndarray, np.ndarray]:
    if verbose:
        print()

    if verbose:
        print(f"Loading MT points from {tubulaton_output_file_path} to numpy array...")
    mt_points, mt_ids = get_mt_points(file_path=tubulaton_output_file_path,
                                      config=config)

    #TODO - It shouldn't really cause issues if no MTs, even though this shouldn't happen if everything went right (But this is a temporary fix)
    if mt_points.shape[0] == 0:
        image = np.zeros((512,512))
        label = np.zeros((512,512))
        return image, label
    #TODO END

    unique_mt_ids = np.unique(mt_ids)

    if mode == 'demo_interactive':
        print("Showing visualization of tubulin (before depoly simulation)")
        visualize(mt_points)

    if verbose:
        print(f"Using depolymerization rate: {depoly_proportion:.2f}...")
    
    #TODO - (mt, tubulin)
    tubulin_points, mt_points = simulate_depoly(mt_points=mt_points,
                                                mt_ids=mt_ids,
                                                config=config,
                                                unique_mt_ids=unique_mt_ids,
                                                proportion=depoly_proportion)
    if mode == 'demo_interactive':
        print("Showing visualization of tubulin (after depoly simulation)")
        visualize(tubulin_points)

    if verbose:
        print("Simulating positions of fluorophores...")
    fluorophore_points = get_fluorophore_points(tubulin_points=tubulin_points,
                                                config=config)
    if mode == 'demo_interactive':
        print("Visualizing fluorophore points...")
        visualize(fluorophore_points,
                  colors='yellow',
                  background_color='black')

    if mode.startswith('demo'):
        plot_rows = 3
        plot_cols = 2
        fig, axs = plt.subplots(plot_rows, plot_cols, figsize=(10 * plot_cols, 2 * plot_rows))
        axs = axs.flatten()
        axs_index = 0
    
    #TODO - It shouldn't be the case no FPs - there are a fixed amount! (But this is a temporary fix)
    if fluorophore_points.shape[0] == 0:
        image = np.zeros((512,512))
        label = np.zeros((512,512))
        return image, label
    #TODO END
    
    #TODO - shouldn't be done like this. Look at the mesh .vtk files! (Do it properly!)
    z_min = fluorophore_points[:, 2].min()
    z_max = fluorophore_points[:, 2].max()
    x_min = fluorophore_points[:, 0].min()
    x_max = fluorophore_points[:, 0].max()
    y_min = fluorophore_points[:, 1].min()
    y_max = fluorophore_points[:, 1].max()

    # TODO - make z_slice tunable (No magic numbers!)
    z_slice=z_min*0.9 + z_max*0.1

    if verbose:
        print("Generating label segmentation...")
    label = get_label(mt_points=mt_points,
                      z_slice=z_slice,
                      config=config,
                      bounding_box=(x_min, y_min, x_max, y_max),
                      verbose=verbose)
        
    if verbose:
        print("Calculating fluorophore density over focal plane...")
    fluorophore_image = get_fluorophore_image(fluorophore_points=fluorophore_points,
                                              config=config,
                                              bounding_box=(x_min, y_min, x_max, y_max),
                                              z_slice=z_slice,
                                              verbose=verbose)
    
    if mode.startswith('demo'):
        axs[axs_index].imshow(fluorophore_image, cmap='gray') #type: ignore
        axs[axs_index].axis('off') #type: ignore
        axs[axs_index].set_title("Fluorophore image") #type: ignore
        axs_index += 1 #type: ignore

    if verbose:
        print("Calculating fluorophore photon emission rate...")
    fluorophore_emission_per_second  = calculate_fluorophore_emission_per_second(config=config)
    if verbose:
        print(f"Fluorophore photon emission per second = {fluorophore_emission_per_second}")
    intensities = fluorophore_image * fluorophore_emission_per_second 

    if verbose:
        print("Applying point spread function to light intensity array...")
    # TODO At present - we can take this (inexpensive) calculation of the kernel out of the loop
    # But eventually want to make the parameters stochastic so the kernel will be recalculated each time
    psf_kernel = get_psf_kernel(config=config)
    intensities = scipy.signal.convolve2d(intensities, psf_kernel, mode='same')

    if mode.startswith('demo'):
        axs[axs_index].imshow(intensities, cmap='gray') #type: ignore
        axs[axs_index].axis('off') #type: ignore
        axs[axs_index].set_title("Fluorophore image after PSF") #type: ignore
        axs_index += 1 #type: ignore

    if verbose:
        print("Simulating shot noise...")
    image = get_digital_signal(intensities=intensities,
                               config=config)
    if mode.startswith('demo'):
        axs[axs_index].imshow(image, cmap='gray') #type: ignore
        axs[axs_index].axis('off') #type: ignore
        axs[axs_index].set_title("Image after Shot Noise") #type: ignore
        axs_index += 1 #type: ignore

    if verbose:
        print("Quantizing intensities...")
    image = quantize_intensities(image=image,
                                 config=config)
    if mode.startswith('demo'):
        axs[axs_index].imshow(image, cmap='gray') #type: ignore
        axs[axs_index].axis('off') #type: ignore
        axs[axs_index].set_title("Image after Quantization") #type: ignore
        axs_index += 1 #type: ignore

    if mode.startswith('demo'):
        axs[axs_index].imshow(label, cmap='gray') #type: ignore
        axs[axs_index].axis('off') #type: ignore
        axs[axs_index].set_title("Label Segmentation") #type: ignore
        axs_index += 1 #type: ignore

    if mode.startswith('demo'):
        colored_label = np.tile(label[:, :, np.newaxis], (1, 1, 4)).astype(np.float32)
        colored_label[:, :, 0] = 1.
        colored_label[:, :, 1] = 0.
        colored_label[:, :, 2] = 0.

        axs[axs_index].imshow(colored_label) #type: ignore
        axs[axs_index].imshow(image, cmap='gray', alpha=0.5) #type: ignore
        axs[axs_index].axis('off') #type: ignore
        axs[axs_index].set_title("Label Segmentation Overlaid") #type: ignore
        axs_index += 1 #type: ignore

    if mode.startswith('demo'):
        plt.tight_layout()
    
    if verbose:
        print()

    return image, label

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

def parse_number_or_range(value : str) -> range:
    try:
        if '-' in value:
            # This is a range like 32-329
            start, end = map(int, value.split('-'))
            if start > end:
                raise ValueError(f"Invalid range: '{value}' (start should be less than or equal to end)")
            return range(start, end + 1)
        else:
            # This is a single number like 23
            num = int(value)
            return range(num, num + 1)
    except ValueError:
        raise ValueError(f"Invalid input format: '{value}' (must be a number or a range in format START-END)")

def parse_args() -> argparse.Namespace:
    # Parse CL arguments
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data from tubulaton .vtk files."
    )

    parser.add_argument('-od', '--output_dir', 
                        type=Path, 
                        help='Output Directory Path\nLeave blank to use value in .env file')

    parser.add_argument('-i', '--input',
                        type=Path,
                        help="Input Path (tubulaton .vtk file or directory of .vtk files)\nLeave blank to use value in .env file")

    parser.add_argument('-ids', '--ids',
                        type=str,
                        help="Which file ids to use? (Either a number (e.g. 32) or a range (e.g. 32-48)")

    parser.add_argument('-c', '--config',
                        type=Path,
                        help='Path to JSON5 config file\nLeave blank to use value in .env file')
    
    parser.add_argument("--depoly",
                        type=float,
                        help="The rate of depolymerization. Leave blank to draw a random value from the distribution specified in config")

    parser.add_argument('--mode', choices=['demo_interactive', 'demo_headless', 'overwrite', 'update'], 
                        default='update',
                        help="""Mode of operation: demo, overwrite, or update (default: update):
                                - demo_interactive: Interactive demonstration of program on a single .vtk file. No file writing.
                                - demo_headless:    Saves (partial) output of demos to files.
                                - overwrite:        Overwrite any existing synthetic data in specified location
                                - update:           Skip over any pre-existing synthetic data files""")

    parser.add_argument('-v', '--verbose', 
                        action='store_true', 
                        help='Increase output verbosity')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    start_time = time.time()
    random.seed(1000)

    args = parse_args()
    load_dotenv()

    mode = args.mode
    verbose = args.verbose

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = Path(os.environ["GENERATE_OUTPUT_DIR"])

    if args.input is not None:
        tubulaton_output_path = args.input
    else:
        tubulaton_output_path = Path(os.environ["TUBULATON_OUTPUT_DIR"])

    if args.config is not None:
        config_file_path = args.config
    else:
        config_file_path = Path(os.environ["GENERATE_CONFIG"])

    file_ids : Optional[range]
    file_ids = parse_number_or_range(args.ids) if (args.ids is not None) else args.ids

    config = load_json5(config_file_path)
    if verbose:
        print(f"Loaded config file: {config_file_path}")

    if args.depoly is None:
        depoly_proportion_distribution_info = config['depoly_proportion_distribution']
        depoly_proportion_distribution  = getattr(np.random, depoly_proportion_distribution_info['name'])
        params = depoly_proportion_distribution_info.get('params', {})
        depoly_proportion = depoly_proportion_distribution(**params)
    else:
        depoly_proportion = args.depoly

    if not mode.startswith('demo'):
        if output_dir is None:
            raise ValueError(f"You must provide an output dir (-o) when not in demo mode!")
        os.makedirs(output_dir / "Images/", exist_ok=True) 
        os.makedirs(output_dir / "Labels/", exist_ok=True) 

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
            raise ValueError(f"Provided [tubulaton_output_path]: {sys.argv[2]} is neither a file nor a directory!")
    
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

    for tubulaton_output_file_path in file_path_iterator:
        block_start_time = time.time()

        file_id = extract_file_id(tubulaton_output_file_path)

        if not mode.startswith('demo'):
            if output_dir is None:
                raise ValueError("Output dir cannot be None when not in demo mode!")

            image_file_name = f"image-{file_id}.png"
            label_file_name = f"label-{file_id}.png"

            image_file_path = output_dir / 'Images' / image_file_name
            label_file_path = output_dir / 'Labels' / label_file_name

            if mode == 'overwrite' or (not image_file_path.exists()) or (not label_file_path.exists()):
                image, label = generate(tubulaton_output_file_path=tubulaton_output_file_path,
                                        depoly_proportion=depoly_proportion,
                                        config=config,
                                        mode=mode,
                                        verbose=verbose)

                image = image * 255. / image.max()
                image = Image.fromarray(image.astype(np.uint8), mode='L')

                label = label * 255
                label = Image.fromarray(label, mode='L')

                image.save(image_file_path)

                label.save(label_file_path)

                if verbose:
                    print(f"Saved data to {image_file_path} and {label_file_path}.")
                    print()
            else:
                if verbose:
                    print(f"Image and label files for {tubulaton_output_file_path} already exist:")
                    print(f"Image path: {image_file_path}")
                    print(f"Label path: {label_file_path}")
                    print(f"Skipping this tubulaton file.")
                    print()
        else:
            image, label = generate(tubulaton_output_file_path=tubulaton_output_file_path,
                                    depoly_proportion=depoly_proportion,
                                    config=config,
                                    mode=mode,
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


    time_taken = time.time() - start_time
    if verbose and mode != 'demo_interactive':
        print(f"\n\nTook {time_taken:.2f} seconds in total.")
        if len(tubulaton_output_file_paths) > 1:
            print(f"({time_taken / len(tubulaton_output_file_paths):.2f} seconds per .vtk file)")

# Interactive demo:
# python3 generate.py -i /Users/karan/MTData/tubulaton-run -c /Users/karan/microsegmentation/generation/generate_config.json5  --mode=demo_interactive -v --id=32

# Headless demo:
# python3 generate.py -i /Users/karan/MTData/tubulaton-run -c /Users/karan/microsegmentation/generation/generate_config.json5  --mode=demo_headless -v --num_files=5

# Run on local computer
# python3 generate.py -o /Users/karan/MTData/Synthetic_TEST -i /Users/karan/MTData/tubulaton-run -c /Users/karan/microsegmentation/generation/generate_config.json5  --mode=update -v 
