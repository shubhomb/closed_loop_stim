#!/bin/bash

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <slurm_script> <config_name> <start_seed> <end_seed>"
  exit 1
fi

SLURM_SCRIPT=$1
CONFIG_NAME=$2
START_SEED=$3
END_SEED=$4

for ((seed=START_SEED; seed<=END_SEED; seed++)); do

  CMD="python -m optlearn.main \
    --config-name $CONFIG_NAME \
    random_seed=${seed}"

  echo "Submitting job with random_seed=${seed}"
  sbatch "$SLURM_SCRIPT" "$CMD"

done