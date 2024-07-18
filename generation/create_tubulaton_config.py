# Command line arguments
# tubulaton_dir  : Path to tubulaton folder 
# input_mesh_dir : Location of folder containing .vtk meshes for membrane/nucleus
# output_dir     : Location to save data
# time_steps     : Number of time steps to simulate
# sample_name    : Unique ID for this particular job (as it is run as an array on SLURM, we don't want processes overwriting eachother!)

import os
import sys
from pathlib import Path

import json5

import numpy as np

def save_config_to_ini(config_dict : dict, file_path : Path) -> None:
    with open(file_path, 'w') as file:
        for key, value in config_dict.items():
            file.write(f"{key}={value}\n")

def generate_config_dict(tubulaton_config : dict,
                         tubulaton_output_dir : Path,
                         input_mesh_dir : Path,
                         num_time_steps : int,
                         output_file_id : str) -> dict:
    config_dict = tubulaton_config['tubulaton_parameters'].copy()

    #TODO Refactor tubulaton to not require the dir path ending in /
    config_dict['nom_folder_output_vtk'] = str(tubulaton_output_dir) + '/'
    config_dict['nom_folder_input_vtk'] = str(input_mesh_dir) + '/'

    config_dict['nom_output_vtk'] = f"tubulaton-{output_file_id}_"

    config_dict['vtk_steps'] = str(num_time_steps)
    config_dict['nb_max_steps'] = str(num_time_steps)

    positive_real_params = config_dict['positive_real_params']
    for param in positive_real_params:
        orig_value = float(config_dict[param])
        sigma = orig_value / 3

        new_value = orig_value + np.random.normal(loc=0, scale=sigma)
        if new_value < 0.:
            new_value = 0.

        config_dict[param] = str(new_value)

    natural_number_params = config_dict['natural_number_params']
    for param in natural_number_params:
        orig_value = float(config_dict[param])
        sigma = orig_value / 3

        new_value = orig_value + np.random.normal(loc=0, scale=sigma)
        if new_value < 0.:
            new_value = 0.
        new_value = int(new_value)

        config_dict[param] = str(new_value)
    
    prob_params = config_dict['prob_params']
    for param in prob_params:
        orig_value = float(config_dict[param])

        new_value = orig_value * (np.random.random() * 2 + 0.3)
        new_value = np.clip(new_value, a_min=0., a_max=1.)

        config_dict[param] = str(new_value)
    
    return config_dict
    
def create_tubulaton_config(tubulaton_dir : Path, 
                            output_dir : Path,
                            input_mesh_dir : Path,
                            tubulaton_config : dict,
                            num_time_steps : int,
                            sample_name : str) -> None:
    tubulaton_output_dir = output_dir / 'tubulaton-run'

    config_dict = generate_config_dict(tubulaton_config=tubulaton_config,
                                       tubulaton_output_dir=tubulaton_output_dir,
                                       input_mesh_dir=input_mesh_dir,
                                       num_time_steps=num_time_steps,
                                       output_file_id=sample_name)

    config_file_path = tubulaton_dir / f"init/config-{sample_name}.ini"

    os.makedirs(tubulaton_output_dir, exist_ok=True)

    save_config_to_ini(config_dict, config_file_path)

if __name__ == '__main__':
    tubulaton_dir = Path(sys.argv[1])
    input_mesh_dir = Path(sys.argv[2])
    tubulaton_config_file_path = Path(sys.argv[3])
    output_dir = Path(sys.argv[4])
    time_steps = int(sys.argv[5])
    sample_name = sys.argv[6]

    with open(tubulaton_config_file_path, 'r') as file:
        tubulaton_config = json5.load(file)
    if not isinstance(tubulaton_config, dict):
        print("Invalid tubulaton_config.json5 file provided!")
        print(f"{tubulaton_config_file_path} is not a dictionary")
        exit(1)

    create_tubulaton_config(tubulaton_dir=tubulaton_dir,
                            output_dir=output_dir,
                            tubulaton_config=tubulaton_config,
                            input_mesh_dir=input_mesh_dir,
                            num_time_steps=time_steps,
                            sample_name=sample_name)
