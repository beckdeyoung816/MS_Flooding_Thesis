#!/bin/bash


source ~/opt/anaconda3/condabin/conda
conda activate py39

declare -a coasts=('NE_Atlantic_Yellow' 'NE_Atlantic_Red' 'NW_Atlantic_Blue' 'Japan_Red')

declare -a losses=('mse' \
                'Gumbel')

for coast in "${!coasts[@]}"; do
    echo "Coast: ${coasts[$coast]}"

    for loss in "${!losses[@]}"; do
        echo "Loss: ${losses[$loss]}"

        # Run the Script
        python -W ignore ML_env_Coast.py ${coasts[$coast]} 'ALL' ${losses[$loss]}
    done
done