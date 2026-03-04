#!/bin/bash
#SBATCH --job-name=aggregate
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Fix MKL unbound variable error
set +u

# Load environment
ml purge
ml Miniforge3
eval "$(conda shell.bash hook)"
conda activate $SHARED_SCRATCH/$USER/envs/patterns5k_3_11

# Restore strict mode after conda activation
set -u

cd ~/patterns_5k

# Parse arguments - default to all sweeps if none provided
if [ $# -eq 0 ]; then
    SWEEP_PATTERN="$SHARED_SCRATCH/$USER/patterns5k_results/slurm_*"
else
    SWEEP_PATTERN="$SHARED_SCRATCH/$USER/patterns5k_results/slurm_${1}_*"
fi

echo "Aggregating sweeps matching: $SWEEP_PATTERN"

# Aggregate the specified sweeps
bash slurm/aggregate_sweeps.sh $SWEEP_PATTERN