#!/bin/bash

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <slurm_script> <config_name> <start_seed> <end_seed>"
  exit 1
fi

SLURM_SCRIPT=$1
CONFIG_NAME=$2
START_SEED=$3
END_SEED=$4

# the specific DOF pairs to test
# DOFS=(0 1 2 3 4) # T5 5finger
# DOFS=(0 1 2 3 4 5) # T16 5finger
DOFS=(0 1 2 3) # T16 202505

# the specific targets & tau pairs to test
TARGETS=(16 32)
# TAU_VALUES=(0.02 0.01) # T5 5finger
TAU_VALUES=(0.03 0.02) # T16 202505

# Loop over the specific DOF pairs
for ((i=0; i<${#DOFS[@]}; i++)); do
  for ((j=i+1; j<${#DOFS[@]}; j++)); do
    for ((k=j+1; k<${#DOFS[@]}; k++)); do
      for ((l=k+1; l<${#DOFS[@]}; l++)); do
        
        DOF1=${DOFS[$i]}
        DOF2=${DOFS[$j]}
        DOF3=${DOFS[$k]}
        DOF4=${DOFS[$l]}

        # Loop over targets and tau pairs
        for ((m=0; m<${#TARGETS[@]}; m++)); do
          target=${TARGETS[$m]}
          tau=${TAU_VALUES[$m]}
        
          # Loop over the seeds
          for ((seed=START_SEED; seed<=END_SEED; seed++)); do

            CMD="python -m optlearn.main \
              --config-name $CONFIG_NAME \
              encoder.use_dof='[${DOF1},${DOF2},${DOF3},${DOF4}]' \
              task.n_targets=${target} \
              task.test_trials=10 \
              sampling.sampler.tau=${tau} \
              random_seed=${seed}"

            echo "Submitting: DOF=[${DOF1},${DOF2},${DOF3},${DOF4}], Targets=${target}, Tau=${tau}, Seed=${seed}"
            sbatch "$SLURM_SCRIPT" "$CMD"
          done
        done
      done
    done
  done
done