#!/usr/bin/env zsh

dataset_dir="/Users/karan/MTData/SimulatedData"
save_file_dir="/Users/karan/Microtubules/classification/models"
save_file_name="model"
patch_size="256"

python3 train.py $dataset_dir $save_file_dir $save_file_name $patch_size
