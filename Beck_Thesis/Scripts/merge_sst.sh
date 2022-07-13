#!/bin/bash
#SBATCH -n 24
#SBATCH -t 0-01:00:00 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=beckdeyoung816@gmail.com

source ~/miniconda3/etc/profile.d/conda.sh
conda activate beck_env 

python merge_sst_data.py

