#!/bin/zsh

source .venv_evaluation/bin/activate

# Define your lists of arguments
models=("heather_s1" "heather_s2" "heather_s3" "heather_s4" "veronica_s1")
datasets=("Synthetic/Eval/Hard" "Synthetic/Eval/HighRes")

# Nested loops to iterate over all combinations of arguments
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        # Call the Python script with the arguments
        python evaluate.py -n "$model" -i "$dataset" -v -c 30
    done
done

