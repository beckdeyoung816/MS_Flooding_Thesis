#!/bin/bash
#SBATCH -n 4 # NUM PROCESSES
#SBATCH -t 0-05:00:00 
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=beckdeyoung@gmail.com

source ~/miniconda3/etc/profile.d/conda.sh
conda activate beck_env

python -W ignore ML_env_Coast.py 'NE_Atlantic_Yellow' 'ANN' 'mse'