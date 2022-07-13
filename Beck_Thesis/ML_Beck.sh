#!/bin/bash
#SBATCH -n 5
#SBATCH -t 0-05:00:00 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=beckdeyoung@gmail.com

source ~/miniconda3/etc/profile.d/conda.sh
conda activate beck_env 

declare -a coasts=('NE_Atlantic_Yellow') #'NE_Atlantic_Red' 'NW_Atlantic_Green' 'Japan_Red')

declare -a losses=('mse')# \
                #'Gumbel')

num_procs=1

declare -a pids=( )

for coast in "${!coasts[@]}"; do
    for loss in "${!losses[@]}"; do
        while (( ${#pids[@]} >= num_procs )); do
            sleep 0.2
            for pid in "${!pids[@]}"; do
                kill -0 "$pid" &>/dev/null || unset "pids[$pid]"
            done
        done
        python -W ignore ML_env_Coast.py ${coasts[$coast]} 'TCN-LSTM' ${losses[$loss]} & pids["$!"]=1
done
done
wait