#!/bin/zsh

# Check if $2 is "overwrite"
if [ "$3" = "overwrite" ]; then
  overwrite_mode="--overwrite"
else
  overwrite_mode=""
fi

if [ "$2" = "" ]; then
  epochs=10
else
  epochs="$2"
fi

# Load .env file
export $(grep -v '^#' .env | xargs)

program_path="$BASE_DIR/segmentation/train.py"

data_dir="$BASE_DIR/Synthetic/"
log_dir="$BASE_DIR/runs/"
model_dir="$BASE_DIR/ModelSaveFiles/"
config_path="$segmentation_dir/config/"
demo_config_path="$segmentation_dir/demo_config.json5"

source $segmentation_dir/.venv_segmentation/bin/activate
python $program_path -c $config_path --name $1 --democonfig $demo_config_path --epochs $epochs -dd $data_dir -ld $log_dir -md $model_dir --verbose $overwrite_mode

#! Notify me the program is done running with sound (or crashed)
while true; do
    afplay /System/Library/Sounds/Glass.aiff
done
