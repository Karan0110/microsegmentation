#!/bin/zsh

BASE_DIR="/Users/karan/microsegmentation"

if [ "$2" = "" ]; then
  epochs_option=""
else
  epochs_option="--epochs $2"
fi

program_path="$BASE_DIR/segmentation/train.py"

source $BASE_DIR/segmentation/.venv_segmentation/bin/activate

# The python program to run
CMD="python $program_path --name $1 $epochs_option --verbose"

echo "Running caffeinated command: $CMD"

# Use caffeinate to prevent idling (e.g. on an overnight run)
caffeinate -i $CMD

#! Signal the program is done running with sound
# while true; do
#     afplay /System/Library/Sounds/Glass.aiff
# done
