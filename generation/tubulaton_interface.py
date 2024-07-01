import subprocess
import os
import shutil
import sys

import numpy as np

from tubulaton_config import default_config_dict 

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
                         num_time_steps : int) -> dict:
    config_dict = default_config_dict.copy()

    config_dict['nom_folder_output_vtk'] = tubulaton_dir
    config_dict['nom_folder_input_vtk'] = input_mesh_dir
    config_dict['nom_input_vtk'] = input_vtk_1
    config_dict['nom_input_vtk_2'] = input_vtk_2
    config_dict['nom_input_vtk_ref'] = input_vtk_ref
    config_dict['nom_input_vtk_ref_2'] = input_vtk_ref

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

def save_tubulaton_vtk(exec_dir : str, 
                       exec_file_name : str,
                       output_dir : str,
                       default_config_dict : dict,
                       input_mesh_dir : str,
                       num_time_steps : int,
                       input_vtk_1 : str = "Cylinder_10000_EqualNucleation.vtk",
                       input_vtk_2 : str = "Sphere_Rad500_500_500_Xpos3625_Mesh2000.vtk",
                       input_vtk_ref : str = "Cylinder_10000_EqualNucleation.vtk",
                       output_file_name : str = 'tubulaton_raw.vtk',
                       verbose : bool = False) -> None:

    if verbose:
        print("Running tubulaton...")
    
    tubulaton_dir = os.path.join(output_dir, "tubulaton-run/")

    config_dict = generate_config_dict(default_config_dict=default_config_dict,
                                       tubulaton_dir=tubulaton_dir,
                                       input_mesh_dir=input_mesh_dir,
                                       input_vtk_1=input_vtk_1,
                                       input_vtk_2=input_vtk_2,
                                       input_vtk_ref=input_vtk_ref,
                                       num_time_steps=num_time_steps)

    tubulaton_file_name = f"{config_dict['nom_output_vtk']}{config_dict['nb_max_steps']}.vtk"
    config_file_path = os.path.join(exec_dir, "config.ini")

    os.makedirs(tubulaton_dir, exist_ok=True)

    save_config_to_ini(config_dict, config_file_path)
    
    os.chdir(exec_dir)
    result = subprocess.run([f'./{exec_file_name}', config_file_path], 
                            capture_output=True, 
                            text=True)
    if result.stderr:
        raise RuntimeError(f"Tubulaton error:\n{result.stderr}")

    shutil.move(os.path.join(tubulaton_dir, tubulaton_file_name), 
                os.path.join(output_dir, output_file_name))
    shutil.rmtree(tubulaton_dir)
    
#Test: Check /Users/karan/Microtubules/Data/Tubulaton-Debug/tubulaton_raw.vtk
# if __name__ == '__main__':
#     print("Creating tubulaton vtk output...")
#     save_tubulaton_vtk(exec_dir="/Users/karan/Microtubules/tubulaton/bin",
#                        exec_file_name='program',
#                        output_dir='/Users/karan/Microtubules/Data/Tubulaton-Debug')
#     print("Saved to file!")
