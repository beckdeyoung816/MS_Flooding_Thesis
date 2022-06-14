#!/bin/bash

conda activate py39

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
  python -W ignore Workshop/ML_env_Beck.py "$station" & pids["$!"]=1
done
wait

python -W ignore Workshop/ML_env_Beck.py 'cuxhaven-cuxhaven-germany-bsh' 'ANN' 'gumbel'