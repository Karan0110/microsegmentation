#!/bin/bash

# Check if an argument is supplied
if [ $# -lt 2 ]; then
    echo "At least 2 arguments needed!"
	echo "Correct usages $0 [sample_name] [time_steps] <environment>"
    echo "Where <environment> is \"hpc\" or left blank"
    exit 1
fi

environment="$3"

if [[ "$environment" == "hpc" ]]; then
    base_dir="/rds/user/ke330/hpc-work"

    tubulaton_dir="$base_dir/tubulaton/"
    input_mesh_dir="$base_dir/tubulaton/structures/2024_Karan/"
    output_dir="$base_dir/SimulatedData/"

    create_tubulaton_config_path="$base_dir/generation/create_tubulaton_config.py"
	meta_config_file_path="$base_dir/generation/tubulaton_config.json5"
    generate_path="$base_dir/generation/generate.py"

	exec_file_name="programme"
else
    tubulaton_dir="/Users/karan/tubulaton/"
    input_mesh_dir="/Users/karan/tubulaton/structures/2024_Karan/"
    output_dir="/Users/karan/MTData/SimulatedData_Test/"

    create_tubulaton_config_path="create_tubulaton_config.py"
	meta_config_file_path="tubulaton_config.json5"
    generate_path="generate.py"

	exec_file_name="tubulaton"
fi

sample_name="$1"
time_steps="$2"

#! File paths/names computed from variables already defined

config_file_name="config-$sample_name.ini"

tubulaton_output_dir="${output_dir}tubulaton-run/"

old_tubulaton_output_file_name="tubulaton-${sample_name}_${time_steps}.vtk"
old_tubulaton_output_file_path="${tubulaton_output_dir}${old_tubulaton_output_file_name}"

tubulaton_output_file_name="tubulaton-${sample_name}.vtk"
tubulaton_output_file_path="${tubulaton_output_dir}${tubulaton_output_file_name}"

#! The main program
#! =================

echo "Setting up config file..."
echo "========================="
python3 $generate_tubulaton_config_path $tubulaton_dir $meta_config_file_path $input_mesh_dir $output_dir $time_steps $sample_name

echo "\n\n"
echo "Running tubulaton..."
echo "===================="

#! Ensure output directory for tubulaton exists
mkdir -p $tubulaton_output_dir

#! Run tubulaton executable
${tubulaton_dir}bin/$exec_file_name ${tubulaton_dir}init/${config_file_name}

#! Now the tubulaton output vtk file name only depends on $sample_name
mv $old_tubulaton_output_file_path $tubulaton_output_file_path

echo "\n\n"
echo "Generating processed training sample..."
echo "======================================="

#! When working on the HPC cluster - we have to unload VTK library to
#! avoid naming conflicts with python vtk module
if [[ "$environment" == "hpc" ]]; then
	module unload vtk/7.1.1
fi

python3 $generate_path $output_dir $sample_name $tubulaton_output_file_path

