import os
import subprocess
import shutil
import argparse
from dotenv import load_dotenv
from pathlib import Path

import json5

from create_tubulaton_config import create_tubulaton_config

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a tubulaton .ini config file"
    )

    parser.add_argument('-c', '--config',
                        type=str,
                        required=True,
                        help="Name of config file")

    parser.add_argument('-n', '--name',
                        type=str,
                        required=True,
                        help="Name of sample")

    parser.add_argument("-v", '--verbose',
                        action='store_true',
                        help='Increase verbosity of output')
    
    return parser.parse_args()

if __name__ == '__main__':
    load_dotenv('../.env')
    args = parse_args()    

    verbose = args.verbose

    tubulaton_dir = Path(os.environ['TUBULATON_DIR'])
    input_mesh_dir = tubulaton_dir / Path(os.environ['TUBULATON_INPUT_MESH_DIR'])

    sample_name = args.name

    tubulaton_configs_dir = Path(os.environ['TUBULATON_CONFIGS_DIR'])
    tubulaton_config_file_name = args.config
    tubulaton_config_file_path = tubulaton_configs_dir / f"{tubulaton_config_file_name}.json5"

    with open(tubulaton_config_file_path, 'r') as file:
        tubulaton_config = json5.load(file)
    if not isinstance(tubulaton_config, dict):
        print("Invalid tubulaton_config.json5 file provided!")
        print(f"{tubulaton_config_file_path} is not a dictionary")
        exit(1)

    dataset_name = tubulaton_config['dataset_name']
    output_dir = Path(os.environ['TUBULATON_OUTPUT_DIR']) / dataset_name

    if verbose:
        print(f"Creating tubulaton .ini config...")
    create_tubulaton_config(tubulaton_dir=tubulaton_dir,
                            output_dir=output_dir,
                            tubulaton_config=tubulaton_config,
                            input_mesh_dir=input_mesh_dir,
                            sample_name=sample_name)
    
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("Running tubulaton...")
    ini_config_file_path = tubulaton_dir / f"init/config-{sample_name}.ini"
    subprocess.run(str((tubulaton_dir / "bin" / "tubulaton").absolute()) + " " + str(ini_config_file_path.absolute()), shell=True)

    time_steps = tubulaton_config['time_steps']
    old_path = output_dir / "tubulaton-run" / f"tubulaton-{sample_name}_{time_steps}.vtk"
    new_path = output_dir / f"tubulaton-{sample_name}"
    if verbose:
        print(f"\nMoving output from {old_path} to {new_path}")
    shutil.move(old_path, new_path)
