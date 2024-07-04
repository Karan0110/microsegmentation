#!/usr/bin/env zsh

dataset_dir="/Users/karan/MTData/SimulatedData"
save_file_path="/Users/karan/Microtubules/classification/models/model.pth"
patch_size="256"

python3 train.py $dataset_dir $save_file_path $patch_size

