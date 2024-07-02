# Command line arguments
# output_dir     : Location to save data
# sample_name    : A unique ID for the data sample

import os
import sys
import numpy as np
import time

from PIL import Image

from vtk_projection import get_2d_projection_from_vtk
from process_projection import process_projection

VERBOSE = True

if __name__ == '__main__':
    start_time = time.time()

    if VERBOSE:
        print("\n\n")

    output_dir = sys.argv[1]
    sample_name = sys.argv[2]

    # After tubulaton has been run by a shell script:

    tubulaton_file_path = None
    with open(f'tubulaton_file_path_{sample_name}.txt', 'r') as file:
        tubulaton_file_path = file.read()

    image = get_2d_projection_from_vtk(vtk_file_path=tubulaton_file_path,
                                       verbose=VERBOSE)

    noised_control_image, noised_depoly_image = process_projection(control_image=image, 
                                                                   verbose=VERBOSE)

    if VERBOSE:
        print("Converting np array to PIL.Image...")

    noised_control_image = Image.fromarray((noised_control_image * 255.).astype(np.uint8), mode='L')
    noised_depoly_image = Image.fromarray((noised_depoly_image * 255.).astype(np.uint8), mode='L')

    os.makedirs(os.path.join(output_dir, 'Control'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'Depoly'), exist_ok=True)

    if VERBOSE:
        print("Saving sample to files...")

    control_file_path = os.path.join(output_dir, f'Control/image-{sample_name}.png')
    depoly_file_path = os.path.join(output_dir, f'Depoly/image-{sample_name}.png')

    noised_control_image.save(control_file_path)
    noised_depoly_image.save(depoly_file_path)

    if VERBOSE:
        print("Saved data to following locations:")
        print(f"Control: {control_file_path}")
        print(f"Depolymerised: {depoly_file_path}")

        time_taken = time.time() - start_time
        print(f"Took {time_taken:.2f} seconds.")
    
