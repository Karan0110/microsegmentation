#!/bin/zsh

# Check if a command-line argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <sample name>"
  exit 1
fi

# Define the Python script path
python_script="/Users/karan/Microtubules/generate_simulated_data.py"

# Define the command-line arguments
exec_dir="/Users/karan/Microtubules/tubulaton/bin"
exec_file_name="programme"
input_mesh_dir="/Users/karan/Microtubules/tubulaton/structures/2024_Karan/"
output_dir="/Users/karan/Microtubules/Data/SimulatedData"
sample_name="$1"

# Run the Python script with the specified arguments
python3 $python_script $exec_dir $exec_file_name $input_mesh_dir $output_dir $sample_name

