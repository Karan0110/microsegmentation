#!/bin/bash

# Check if an argument is supplied
if [ $# -lt 2 ]; then
    echo "At least 2 arguments needed!"
	echo "Correct usages $0 [sample_name] [time_steps] <environment>"
    echo "Where <environment> is \"hpc\" or left blank"
    exit 1
fi

#! Path variables - defined depending on whether we are working on HPC cluster
#! personal computer
environment="$3"

if [[ "$environment" == "hpc" ]]; then
    base_dir="/rds/user/ke330/hpc-work"

    tubulaton_dir="$base_dir/tubulaton/"
    input_mesh_dir="$base_dir/tubulaton/structures/2024_Karan/"
    output_dir="$base_dir/SimulatedData/"

    config_setup_path="$base_dir/generation/config_setup.py"
    generate_path="$base_dir/generation/generate.py"
else
    tubulaton_dir="/Users/karan/tubulaton/"
    input_mesh_dir="/Users/karan/tubulaton/structures/2024_Karan/"
    output_dir="/Users/karan/MTData/Test_Output/"

    config_setup_path="config_setup.py"
    generate_path="generate.py"
fi

sample_name="$1"
time_steps="$2"

#! File paths/names computed from variables already defined
config_file_name="config-$sample_name.ini"

tubulaton_output_dir="${output_dir}tubulaton-run/"

old_tubulaton_output_file_name="tubulaton-${sample_name}_${time_steps}.vtk"
old_tubulaton_output_file_path="${tubulaton_output_dir}${old_tubulaton_output_file_name}"

new_tubulaton_output_file_name="tubulaton-${sample_name}.vtk"
new_tubulaton_output_file_path="${tubulaton_output_dir}${new_tubulaton_output_file_name}"

#! The main program
#! =================

echo "Setting up config file..."
echo "========================="
python3 $config_setup_path $tubulaton_dir $input_mesh_dir $output_dir $time_steps $sample_name

#! Ensure output directory for tubulaton exists
mkdir -p $tubulaton_output_dir

echo "\n\n"
echo "Running tubulaton..."
echo "===================="
${tubulaton_dir}bin/tubulaton ${tubulaton_dir}init/${config_file_name}

#! Now the tubulaton output vtk file name only depends on $sample_name
mv $old_tubulaton_output_file_path $new_tubulaton_output_file_path

#! When working on the HPC cluster - we have to unload VTK library to
#! avoid naming conflicts with python vtk module
if [[ "$environment" == "hpc" ]]; then
    module unload vtk/7.1.1
fi

echo "\n\n"
echo "Generating processed training sample..."
echo "======================================="

python3 $generate_path $output_dir $sample_name $tubulaton_output_file_path

