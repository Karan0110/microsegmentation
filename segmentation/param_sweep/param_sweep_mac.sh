#!/bin/zsh

export $(grep -v '^#' ../.env | xargs)

script_path="$BASE_DIR/segmentation/train.py"

# Path to the folder (don't include / at the end)
folder_path="$BASE_DIR/segmentation/param_sweep/configs"

while true; do
    # Iterate through the files in the folder
    for filepath in "$folder_path"/*; do
        # Call the function with the filename as a parameter
        local filename=$(basename "$filepath")

        filename_without_extension="${filename%.json5}"

        echo "Training ${filename}..."

        python $script_path --config $filepath --name $filename_without_extension --verbose --epochs 3
    done
done
