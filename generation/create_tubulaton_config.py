import os
from pathlib import Path

def save_config_to_ini(config_dict : dict, file_path : Path) -> None:
    with open(file_path, 'w') as file:
        for key, value in config_dict.items():
            file.write(f"{key}={value}\n")

def generate_config_dict(tubulaton_config : dict,
                         tubulaton_output_dir : Path,
                         input_mesh_dir : Path,
                         output_file_id : str) -> dict:
    config_dict = tubulaton_config.copy()
    tubulaton_parameters = config_dict['tubulaton_parameters']

    #TODO Refactor tubulaton to not require the dir path ending in /
    tubulaton_parameters['nom_folder_output_vtk'] = str(tubulaton_output_dir) + '/'
    tubulaton_parameters['nom_folder_input_vtk'] = str(input_mesh_dir) + '/'

    tubulaton_parameters['nom_output_vtk'] = f"tubulaton-{output_file_id}_"

    num_time_steps = tubulaton_config['time_steps']
    tubulaton_parameters['vtk_steps'] = str(num_time_steps)
    tubulaton_parameters['nb_max_steps'] = str(num_time_steps)

    return tubulaton_parameters
    
def create_tubulaton_config(tubulaton_dir : Path, 
                            output_dir : Path,
                            input_mesh_dir : Path,
                            tubulaton_config : dict,
                            sample_name : str) -> None:
    tubulaton_output_dir = output_dir / 'tubulaton-run'

    config_dict = generate_config_dict(tubulaton_config=tubulaton_config,
                                       tubulaton_output_dir=tubulaton_output_dir,
                                       input_mesh_dir=input_mesh_dir,
                                       output_file_id=sample_name)

    config_file_path = tubulaton_dir / f"init/config-{sample_name}.ini"

    os.makedirs(tubulaton_output_dir, exist_ok=True)

    save_config_to_ini(config_dict, config_file_path)
