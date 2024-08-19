#!/bin/zsh
# Run multiple instances of train.py together, using different config files (e.g. for a parameter sweep)
# NOTE: This is not using parallel processing - the script just alternates between different models every 3 epochs

BASE_DIR="/Users/karan/microsegmentation"
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
