# Command line arguments
# exec_dir       : Path to bin folder for tubulaton
# exec_file_name : Name of tubulaton executable (should be programme)
# input_mesh_dir : Location of folder containing .vtk meshes for membrane/nucleus
# output_dir     : Location to save data
# sample_name    : A unique ID for the data sample

import os
import sys
import numpy as np
import time

from PIL import Image

from tubulaton_interface import save_tubulaton_vtk
from vtk_projection import get_2d_projection_from_vtk
from process_projection import process_projection
from tubulaton_config import default_config_dict

VERBOSE = True
TIME_STEPS = 1000

if __name__ == '__main__':
    start_time = time.time()

    exec_dir = sys.argv[1]
    exec_file_name = sys.argv[2]
    input_mesh_dir = sys.argv[3]
    output_dir = sys.argv[4]
    sample_name = sys.argv[5]

    #This can be anything - just make sure it is consistent between save_tubulaton_vtk and get_2d_projection_from_vtk
    vtk_file_name = 'tubulaton_raw.vtk'

    save_tubulaton_vtk(exec_dir=exec_dir,
                       exec_file_name=exec_file_name,
                       input_mesh_dir=input_mesh_dir,
                       output_dir=output_dir,
                       output_file_name=vtk_file_name,
                       default_config_dict=default_config_dict,
                       num_time_steps=TIME_STEPS,
                       verbose=VERBOSE)


    image = get_2d_projection_from_vtk(vtk_file_path=os.path.join(output_dir, vtk_file_name),
                                                verbose=VERBOSE)
    print(image.shape)


    noised_control_image, noised_depoly_image = process_projection(control_image=image)
        

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
    