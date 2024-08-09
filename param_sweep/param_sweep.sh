#!/bin/zsh

base_dir="/Users/karan/microsegmentation"
script_path="$base_dir/segmentation/train.py"
data_dir="/Users/karan/MTData/Synthetic/"
log_dir="$base_dir/runs/"
demo_config_path="$base_dir/segmentation/demo-config.json5"
model_dir="$base_dir/ModelSaveFiles/"

# Path to the folder (don't include / at the end)
folder_path="$base_dir/config_files_test"

while true; do
    # Iterate through the files in the folder
    for filepath in "$folder_path"/*; do
        # Call the function with the filename as a parameter
        local filename=$(basename "$filepath")

        filename_without_extension="${filename%.json5}"

        echo "Running $filename"

        python $script_path -c $filepath -dc $demo_config_path --name $filename_without_extension -dd $data_dir -ld $log_dir -md $model_dir --verbose --epochs 3
    done
done
