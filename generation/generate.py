# Command line arguments
# output_dir                 : Location to save data
# sample_name                : A unique ID for the data sample
# tubulaton_output_file_path : Path of tubulaton .vtk output file

import os
import sys
import json5
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.signal

from PIL import Image

from fluorophore_simulation import get_mt_points, get_fluorophore_points, get_fluorophore_image, calculate_fluorophore_emission_per_second
from microscope_simulation import get_psf_kernel, get_digital_signal, adjust_contrast, quantize_intensities
from depoly_simulation import simulate_depoly

VERBOSE = True

def generate_from_mt_points(mt_points : np.ndarray,
                            config : dict,
                            verbose : bool) -> np.ndarray:
    if verbose:
        print("Simulating positions of fluorophores...")
    fluorophore_points = get_fluorophore_points(mt_points=mt_points,
                                                config=config)

    # TODO - make z_slice tunable (i.e. actually use the z_slice argument)
    if verbose:
        print("Calculating fluorophore density over focal plane...")
    z_min = fluorophore_points[:, 2].min()
    z_max = fluorophore_points[:, 2].max()

    fluorophore_image = get_fluorophore_image(fluorophore_points=fluorophore_points,
                                              config=config,
                                              z_slice=z_min*0.9 + z_max*0.1,
                                              verbose=VERBOSE)

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

    if verbose:
        print("Simulating shot noise...")
    image = get_digital_signal(intensities=intensities,
                               config=config)

    # image = adjust_contrast(image=image, quantile=adjust_contrast_quantile)

    if verbose:
        print("Quantizing intensities...")
    image = quantize_intensities(image=image,
                                 config=config)

    return image

if __name__ == '__main__':
    start_time = time.time()

    if len(sys.argv) != 4:
        print("Invalid command line arguments!")
        print("Correct usage of program:")
        print(f"python3 {sys.argv[0]} [output_dir] [tubulaton_output_path] [config_file_path]")
        print("Where [tubulaton_output_path] can be a file or directory.")
        exit(1)

    output_dir : Path = Path(sys.argv[1])
    tubulaton_output_path : Path = Path(sys.argv[2])
    config_file_path : Path = Path(sys.argv[3])

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

    os.makedirs(os.path.join(output_dir, 'Control'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'Depoly'), exist_ok=True)

    for output_index, tubulaton_output_file_path in enumerate(tubulaton_output_file_paths):
        if VERBOSE:
            print(f"Generating synthetic data from file {tubulaton_output_file_path}:")
            print()

        if VERBOSE:
            print("Loading MT points from .vtk file to numpy array...")
        mt_points, mt_ids = get_mt_points(file_path=tubulaton_output_file_path,
                                          config=config)

        if VERBOSE:
            print("Simulating depolymerization...")
        depoly_mt_points = simulate_depoly(mt_points=mt_points,
                                           config=config)

        if VERBOSE:
            print("Generating control image...")
        control_image = generate_from_mt_points(mt_points=mt_points,
                                                config=config,
                                                verbose=VERBOSE)

        if VERBOSE:
            print("Generating depolymerized image...")
        depoly_image = generate_from_mt_points(mt_points=depoly_mt_points,
                                               config=config,
                                               verbose=VERBOSE)

        if VERBOSE:
            print("Converting np array to PIL.Image...")

        control_image = control_image * 255. / control_image.max()
        depoly_image = depoly_image * 255. / depoly_image.max()

        control_image = Image.fromarray(control_image.astype(np.uint8), mode='L')
        depoly_image = Image.fromarray(depoly_image.astype(np.uint8), mode='L')

        if VERBOSE:
            print("Saving synthetic data to files...")

        control_file_path = output_dir / f'Control/image-{output_index+1}.png'
        depoly_file_path = output_dir / f'Depoly/image-{output_index+1}.png'

        control_image.save(control_file_path)
        depoly_image.save(depoly_file_path)

        if VERBOSE:
            print()
            print("Saved data to following locations:")
            print(f"Control: {control_file_path}")
            print(f"Depolymerised: {depoly_file_path}")

    time_taken = time.time() - start_time
    if VERBOSE:
        print(f"Took {time_taken:.2f} seconds.")

# output_dir :            /Users/karan/MTData/Test_Output
# tubulaton_output_path : /Users/karan/MTData/tubulaton-run
# config_file_path :      /Users/karan/Microtubules/generation/config.json5
# python3 generate.py /Users/karan/MTData/Test_Output /Users/karan/MTData/tubulaton-run/ /Users/karan/Microtubules/generation/config.json5
