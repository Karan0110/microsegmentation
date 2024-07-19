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
from typing import Iterable, Tuple

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

VERBOSE = True
DEMO_MODE = False

def generate(tubulaton_output_file_path : Path,
             config : dict,
             depoly_proportion : float,
             verbose : bool) -> Tuple[np.ndarray, np.ndarray]:
    if verbose:
        print(f"Loading MT points from {tubulaton_output_file_path} to numpy array...")
    mt_points, mt_ids = get_mt_points(file_path=tubulaton_output_file_path,
                                      config=config)
    unique_mt_ids = np.unique(mt_ids)

    if DEMO_MODE:
        print("Showing visualization of tubulin (before depoly simulation)")
        visualize(mt_points)

    if verbose:
        print(f"Simulating depolymerization (at rate {depoly_proportion})...")
    
    #TODO - (mt, tubulin)
    tubulin_points, mt_points = simulate_depoly(mt_points=mt_points,
                                                mt_ids=mt_ids,
                                                config=config,
                                                unique_mt_ids=unique_mt_ids,
                                                proportion=depoly_proportion)
    if DEMO_MODE:
        print("Showing visualization of tubulin (after depoly simulation)")
        visualize(tubulin_points)

    if verbose:
        print("Simulating positions of fluorophores...")
    fluorophore_points = get_fluorophore_points(tubulin_points=tubulin_points,
                                                config=config)
    if DEMO_MODE:
        print("Visualizing fluorophore points...")
        visualize(fluorophore_points,
                  colors='yellow',
                  background_color='black')

    if DEMO_MODE:
        plot_rows = 3
        plot_cols = 2
        fig, axs = plt.subplots(plot_rows, plot_cols, figsize=(10 * plot_cols, 2 * plot_rows))
        axs = axs.flatten()
    
    z_min = fluorophore_points[:, 2].min()
    z_max = fluorophore_points[:, 2].max()
    x_min = fluorophore_points[:, 0].min()
    x_max = fluorophore_points[:, 0].max()
    y_min = fluorophore_points[:, 1].min()
    y_max = fluorophore_points[:, 1].max()

    # TODO - magic number
    z_slice=z_min*0.9 + z_max*0.1

    if verbose:
        print("Generating label segmentation...")
    label = get_label(mt_points=mt_points,
                      z_slice=z_slice,
                      config=config,
                      bounding_box=(x_min, y_min, x_max, y_max),
                      verbose=VERBOSE)
        
    # TODO - make z_slice tunable (i.e. actually use the z_slice argument)
    if verbose:
        print("Calculating fluorophore density over focal plane...")
    fluorophore_image = get_fluorophore_image(fluorophore_points=fluorophore_points,
                                              config=config,
                                              bounding_box=(x_min, y_min, x_max, y_max),
                                              z_slice=z_slice,
                                              verbose=VERBOSE)
    
    if DEMO_MODE:
        axs[0].imshow(fluorophore_image, cmap='gray') #type: ignore
        axs[0].axis('off') #type: ignore
        axs[0].set_title("Fluorophore image") #type: ignore

    if verbose:
        print("Calculating fluorophore photon emission rate...")
    fluorophore_emission_per_second  = calculate_fluorophore_emission_per_second(config=config)
    if verbose:
        print(f"Fluorophore photon emission per second = {fluorophore_emission_per_second}")
    intensities = fluorophore_image * fluorophore_emission_per_second 

    if verbose:
        print("Applying point spread function to light intensity array...")
    # At present - we can take this (inexpensive) calculation of the kernel out of the loop
    # But eventually want to make the parameters stochastic so the kernel will be recalculated each time
    psf_kernel = get_psf_kernel(config=config)
    intensities = scipy.signal.convolve2d(intensities, psf_kernel, mode='same')

    if DEMO_MODE:
        axs[1].imshow(intensities, cmap='gray') #type: ignore
        axs[1].axis('off') #type: ignore
        axs[1].set_title("Fluorophore image after PSF") #type: ignore

    if verbose:
        print("Simulating shot noise...")
    image = get_digital_signal(intensities=intensities,
                               config=config)
    if DEMO_MODE:
        axs[2].imshow(image, cmap='gray') #type: ignore
        axs[2].axis('off') #type: ignore
        axs[2].set_title("Image after Shot Noise") #type: ignore

    if verbose:
        print("Quantizing intensities...")
    image = quantize_intensities(image=image,
                                 config=config)
    if DEMO_MODE:
        axs[3].imshow(image, cmap='gray') #type: ignore
        axs[3].axis('off') #type: ignore
        axs[3].set_title("Image after Quantization") #type: ignore

    if DEMO_MODE:
        axs[4].imshow(label, cmap='gray') #type: ignore
        axs[4].axis('off') #type: ignore
        axs[4].set_title("Label Segmentation") #type: ignore

    if DEMO_MODE:
        coloured_label = np.tile(label[:, :, np.newaxis], (1, 1, 4)).astype(np.float32)
        coloured_label[:, :, 1] = 0.
        coloured_label[:, :, 2] = 0.
        for i in range(coloured_label.shape[0]):
            for j in range(coloured_label.shape[1]):
                if coloured_label[i,j,0] == 0.:
                    coloured_label[i,j,3] = 0.

        axs[5].imshow(coloured_label) #type: ignore

        axs[5].imshow(image, cmap='gray', alpha=0.5) #type: ignore
        axs[5].axis('off') #type: ignore
        axs[5].set_title("Label Segmentation Overlaid") #type: ignore

    if DEMO_MODE:
        plt.tight_layout()
        plt.show()

    return image, label

if __name__ == '__main__':
    start_time = time.time()
    random.seed(1000)

    if len(sys.argv) not in [4,5]:
        print("Invalid command line arguments!")
        print("Correct usage of program:")
        print(f"python3 {sys.argv[0]} [output_dir] [tubulaton_output_path] [config_file_path] <depoly_rate>")
        print("Where [tubulaton_output_path] can be a file or directory.")
        print("And if [output_dir] is \"DEMO\" program is run in demo mode, not saving any outputs to disk")
        exit(1)

    if sys.argv[1].lower() == 'demo':
        DEMO_MODE = True
    else:
        output_dir : Path = Path(sys.argv[1])
        DEMO_MODE = False
    tubulaton_output_path : Path = Path(sys.argv[2])
    config_file_path : Path = Path(sys.argv[3])

    if len(sys.argv) <= 4:
        depoly_proportion = np.random.random()
    else:
        depoly_proportion = float(sys.argv[4])

    if not DEMO_MODE:
        os.makedirs(output_dir / "Images/", exist_ok=True) #type: ignore
        os.makedirs(output_dir / "Labels/", exist_ok=True) #type: ignore

    tubulaton_output_file_paths : list
    if tubulaton_output_path.is_dir():
        tubulaton_output_file_paths = [file_path for file_path in tubulaton_output_path.iterdir() if file_path.suffix == '.vtk'] 
    elif tubulaton_output_path.is_file():
        tubulaton_output_file_paths = [tubulaton_output_path]
    else:
        print(f"Provided [tubulaton_output_path]: {sys.argv[2]} is neither a file nor a directory!")
        exit(1)

    if VERBOSE:
        print("Loading config file...")
    with open(config_file_path, 'r') as file:
        config = json5.load(file)
    if not isinstance(config, dict):
        raise TypeError("JSON5 config file is of invalid format! It should be a dictionary")

    file_path_iterator : Iterable   
    if DEMO_MODE:
        file_path = random.choice(tubulaton_output_file_paths)
        print(f"Demo file choice: {file_path}")
        file_path_iterator = enumerate([file_path])
    else:
        file_path_iterator = enumerate(tubulaton_output_file_paths)

    for output_index, tubulaton_output_file_path in file_path_iterator:
        # TODO - Magic number! (depoly_proportion argument)
        image, label = generate(tubulaton_output_file_path=tubulaton_output_file_path,
                                depoly_proportion=depoly_proportion,
                                config=config,
                                verbose=VERBOSE)
        if not DEMO_MODE:
            if VERBOSE:
                print("Converting np arrays to PIL.Image...")

            image = image * 255. / image.max()
            image = Image.fromarray(image.astype(np.uint8), mode='L')

            label = label * 255
            label = Image.fromarray(label, mode='L')

            if VERBOSE:
                print("Saving synthetic data to files...")

            image_file_path = output_dir / f'Images/image-{output_index+1}.png' #type: ignore
            image.save(image_file_path)

            label_file_path = output_dir / f'Labels/label-{output_index+1}.png' #type: ignore
            label.save(label_file_path)

            if VERBOSE:
                print(f"Saved data to {image_file_path} and {label_file_path}.")
                print()

    time_taken = time.time() - start_time
    if VERBOSE and not DEMO_MODE:
        print()
        print(f"Took {time_taken:.2f} seconds.")
        print(f"({time_taken / len(tubulaton_output_file_paths):.2f} seconds per .vtk file)")

# To run the demo:
# ----------------
# python3 generate.py DEMO /Users/karan/MTData/tubulaton-run/ /Users/karan/Microtubules/generation/generate_config.json5 0.3
