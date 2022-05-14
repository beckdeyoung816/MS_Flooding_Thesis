#!/bin/bash
#SBATCH -n 24
#SBATCH -t 5-00:00:00
#SBATCH -p normal
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=timothy.tiggeloven@vu.nl

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hydrology

num_procs=5 #24

declare -A pids=( )

for station in $(seq 0 1 30)
do
  while (( ${#pids[@]} >= num_procs )); do
    sleep 0.2
    for pid in "${!pids[@]}"; do
      kill -0 "$pid" &>/dev/null || unset "pids[$pid]"
    done
  done
#  echo $station
  python -W ignore ML_env.py "$station" & pids["$!"]=1
done
wait
