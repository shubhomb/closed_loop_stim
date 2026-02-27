#!/bin/bash

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <slurm_script> <config_name> <start_seed> <end_seed>"
  exit 1
fi

SLURM_SCRIPT=$1
CONFIG_NAME=$2
START_SEED=$3
END_SEED=$4

# Define the specific tau values to test
# TAU_VALUES=(0.01 0.02 0.03 0.04 0.05 0.06 0.08 0.1 0.2)
TAU_VALUES=(0.006 0.008 0.01 0.02 0.03 0.05)
# the specific targets to test
TARGETS=(16 32)

# Loop over tau values
for tau in "${TAU_VALUES[@]}"; do
  # Loop over the targets
  for target in "${TARGETS[@]}"; do
    # Loop over the seeds
    for ((seed=START_SEED; seed<=END_SEED; seed++)); do
      # T5 5finger DOF pair is [2,3,4]
      # T16 5finger DOF pair is [3,4,5]
      # T16 202505 DOF pair is [0,1,2]
      CMD="python -m optlearn.main \
        --config-name $CONFIG_NAME \
        encoder.use_dof='[0,1,2]' \
        task.n_targets=${target} \
        task.test_trials=10 \
        sampling.sampler.tau=${tau} \
        random_seed=${seed}"

      echo "Submitting: tau=${tau}, target=${target}, Seed=${seed}"
      sbatch "$SLURM_SCRIPT" "$CMD"

    done
  done
done