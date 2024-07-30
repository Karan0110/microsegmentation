#!/bin/bash

# Check if an argument is supplied
if [ $# -lt 1 ]; then
    echo "At least 1 argument needed!"
	echo "Correct usage: $0 [environment] <file_id>"
    echo "Where [environment] is \"hpc\" or \"home\""
    echo "And input .vtk file should have name tubulaton-<file_id>.vtk"
    exit 1
fi

environment="$1"
file_id="$2"

if [[ "$environment" == "hpc" ]]; then
    base_dir="/rds/user/ke330/hpc-work"
    generate_dir="$base_dir/microsegmentation/generation"

    output_dir="$base_dir/SyntheticData/"
    tubulaton_output_path="$base_dir/tubulaton-run"
    config_file_path="$generate_dir/generate_config.json5"

    python_file_path="$generate_dir/generate.py"
else
    output_dir="/Users/karan/MTData/Synthetic_TEST"
    tubulaton_output_path="/Users/karan/MTData/tubulaton-run"
    config_file_path="/Users/karan/microsegmentation/generation/generate_config.json5"

    python_file_path="/Users/karan/microsegmentation/generation/generate.py"
fi

python3 $python_file_path $output_dir $tubulaton_output_path $config_file_path $file_id

