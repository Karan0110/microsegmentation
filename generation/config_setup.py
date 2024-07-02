# Command line arguments
# exec_dir       : Path to bin folder for tubulaton
# input_mesh_dir : Location of folder containing .vtk meshes for membrane/nucleus
# output_dir     : Location to save data
# time_steps     : Number of time steps to simulate
# sample_name    : Unique ID for this particular job (as it is run as an array on SLURM, we don't want processes overwriting eachother!)

import os
import sys

import numpy as np

from tubulaton_config import default_config_dict

VERBOSE = True

def save_config_to_ini(config_dict : dict, file_path : str) -> None:
    with open(file_path, 'w') as file:
        for key, value in config_dict.items():
            file.write(f"{key}={value}\n")

def generate_config_dict(default_config_dict : dict,
                         tubulaton_dir : str,
                         input_mesh_dir : str,
                         input_vtk_1 : str,
                         input_vtk_2 : str,
                         input_vtk_ref : str,
                         num_time_steps : int,
                         output_file_id : str) -> dict:
    config_dict = default_config_dict.copy()

    config_dict['nom_folder_output_vtk'] = tubulaton_dir
    config_dict['nom_folder_input_vtk'] = input_mesh_dir
    config_dict['nom_input_vtk'] = input_vtk_1
    config_dict['nom_input_vtk_2'] = input_vtk_2
    config_dict['nom_input_vtk_ref'] = input_vtk_ref
    config_dict['nom_input_vtk_ref_2'] = input_vtk_ref
    config_dict['nom_output_vtk'] = output_file_id

    config_dict['vtk_steps'] = str(num_time_steps)
    config_dict['nb_max_steps'] = str(num_time_steps)

    positive_real_params = ["Angle_bundle", "Angle_cut", "Angle_mb_limite", "Angle_mb_trajectoire"]
    for param in positive_real_params:
        orig_value = float(config_dict[param])
        sigma = orig_value / 3

        new_value = orig_value + np.random.normal(loc=0, scale=sigma)
        if new_value < 0.:
            new_value = 0.

        config_dict[param] = str(new_value)

    natural_number_params = ["D_bundle", "nb_microtubules_init", "nb_microtubules_init_2"]
    for param in natural_number_params:
        orig_value = float(config_dict[param])
        sigma = orig_value / 3

        new_value = orig_value + np.random.normal(loc=0, scale=sigma)
        if new_value < 0.:
            new_value = 0.
        new_value = int(new_value)

        config_dict[param] = str(new_value)
    
    prob_params = ["proba_crossmicro_cut", "proba_crossmicro_shrink", "proba_detachement_par_step_par_microtubule", "proba_initialisation_area", "proba_initialisation_area_2", 'proba_shrink'] 
    for param in prob_params:
        orig_value = float(config_dict[param])

        new_value = orig_value * (np.random.random() * 2 + 0.3)
        new_value = np.clip(new_value, a_min=0., a_max=1.)

        config_dict[param] = str(new_value)
    
    return config_dict
    
def generate_config(exec_dir : str, 
                    output_dir : str,
                    default_config_dict : dict,
                    input_mesh_dir : str,
                    num_time_steps : int,
                    sample_name : str,
                    input_vtk_1 : str = "Cylinder_10000_EqualNucleation.vtk",
                    input_vtk_2 : str = "Sphere_Rad500_500_500_Xpos3625_Mesh2000.vtk",
                    input_vtk_ref : str = "Cylinder_10000_EqualNucleation.vtk") -> None:
    
    tubulaton_dir = os.path.join(output_dir, "tubulaton-run/")

    config_dict = generate_config_dict(default_config_dict=default_config_dict,
                                       tubulaton_dir=tubulaton_dir,
                                       input_mesh_dir=input_mesh_dir,
                                       input_vtk_1=input_vtk_1,
                                       input_vtk_2=input_vtk_2,
                                       input_vtk_ref=input_vtk_ref,
                                       num_time_steps=num_time_steps,
                                       output_file_id=sample_name)

    tubulaton_file_name = f"{config_dict['nom_output_vtk']}{config_dict['nb_max_steps']}.vtk"
    config_file_path = os.path.join(exec_dir, "config.ini")

    os.makedirs(tubulaton_dir, exist_ok=True)
    save_config_to_ini(config_dict, config_file_path)

    tubulaton_file_path = os.path.join(tubulaton_dir, tubulaton_file_name)
    return tubulaton_file_path


if __name__ == '__main__':
    print("CL args for config_setup.py:")
    print(sys.argv)
    print('\n\n')

    exec_dir = sys.argv[1]
    input_mesh_dir = sys.argv[2]
    output_dir = sys.argv[3]
    time_steps = sys.argv[4]
    sample_name = sys.argv[5]

    tubulaton_file_path = generate_config(exec_dir=exec_dir,
                                          output_dir=output_dir,
                                          default_config_dict=default_config_dict,
                                          input_mesh_dir=input_mesh_dir,
                                          num_time_steps=time_steps,
                                          sample_name=sample_name)

    with open(f'tubulaton_file_path_{sample_name}.txt', 'w') as file:
        file.write(tubulaton_file_path)
