#!/bin/zsh

# Check if $2 is "overwrite"
if [ "$2" = "overwrite" ]; then
  overwrite_mode="--overwrite"
else
  overwrite_mode=""
fi

base_dir="/Users/karan/microsegmentation"
segmentation_dir="$base_dir/segmentation"

program_path="$segmentation_dir/train.py"

data_dir="/Users/karan/MTData/Synthetic/"
log_dir="$segmentation_dir/runs/"
model_dir="$base_dir/Models/"
config_dir="$segmentation_dir/config/"

source $segmentation_dir/.venv_segmentation/bin/activate
python $program_path -c $config_dir --name $1 -dd $data_dir -ld $log_dir -md $model_dir --verbose $overwrite_mode
