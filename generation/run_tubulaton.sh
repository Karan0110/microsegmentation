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
    generation_dir="$base_dir/microsegmentation/generation"

    tubulaton_dir="$base_dir/tubulaton/"
    input_mesh_dir="$base_dir/tubulaton/structures/2024_Karan/"
    output_dir="$base_dir/tubulaton-run/"

    create_tubulaton_config_path="$generation_dir/create_tubulaton_config.py"
    meta_config_file_path="$generation_dir/tubulaton_config.json5"
else
    generation_dir="/Users/karan/microsegmentation/generation"
    tubulaton_dir="/Users/karan/tubulaton/"
    input_mesh_dir="/Users/karan/tubulaton/structures/2024_Karan/"
    output_dir="/Users/karan/MTData/tubulaton-run_TEST/"

    create_tubulaton_config_path="$generation_dir/create_tubulaton_config.py"
    meta_config_file_path="$generation_dir/tubulaton_config.json5"
fi

sample_name="$1"
time_steps="$2"

#! File paths/names computed from variables already defined

config_file_name="config-$sample_name.ini"

old_tubulaton_output_file_name="tubulaton-${sample_name}_${time_steps}.vtk"
old_tubulaton_output_file_path="${output_dir}tubulaton-run/${old_tubulaton_output_file_name}"

tubulaton_output_file_name="tubulaton-${sample_name}.vtk"
tubulaton_output_file_path="${output_dir}${tubulaton_output_file_name}"

#! The main program
#! =================

echo "Setting up config file..."
echo "========================="
python3 $create_tubulaton_config_path $tubulaton_dir $input_mesh_dir $meta_config_file_path $output_dir $time_steps $sample_name

echo -e "\n\n"
echo "Running tubulaton..."
echo "===================="

#! Ensure output directory for tubulaton exists
mkdir -p $output_dir

#! Run tubulaton executable
${tubulaton_dir}bin/tubulaton ${tubulaton_dir}init/${config_file_name}

echo -e "\n"

#! Now the tubulaton output vtk file name only depends on $sample_name
mv $old_tubulaton_output_file_path $tubulaton_output_file_path

echo "Output .vtk file saved to $tubulaton_output_file_path"

#! When working on the HPC cluster - we have to unload VTK library to
#! avoid naming conflicts with python vtk module
# if [[ "$environment" == "hpc" ]]; then
# 	module unload vtk/7.1.1
# fi
