#!/bin/zsh

base_dir="/Users/karan/microsegmentation"
script_path="$base_dir/segmentation/train.py"
data_dir="/Users/karan/MTData/Synthetic/"
log_dir="$base_dir/runs/"
model_dir="$base_dir/ModelSaveFiles/"

# Path to the folder (don't include / at the end)
folder_path="$base_dir/param_sweep/config_files"

# Iterate through the files in the folder
for filepath in "$folder_path"/*; do
    # Call the function with the filename as a parameter
    local filename=$(basename "$filepath")

    echo "Running $filename"

    python $script_path -c $filepath --name $filename -dd $data_dir -ld $log_dir -md $model_dir --verbose --overwrite
done
