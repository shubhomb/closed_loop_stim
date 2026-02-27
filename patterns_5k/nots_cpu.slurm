#!/bin/bash
# ===========================================================================
# SLURM job template for NOTS (Rice CRC) — CPU-only node
#
# Usage:  sbatch nots_cpu.slurm "python slurm_experiment.py --config ..."
#
# Lighter-weight than the GPU template; suitable for small models or
# if GPU queues are congested.
# ===========================================================================
#SBATCH --job-name=patterns5k
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/patterns5k_%j.out
#SBATCH --error=slurm_logs/patterns5k_%j.err
#SBATCH --mail-type=FAIL

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ---- Load environment ----
# Use Miniforge (conda-forge) to reproduce the local 3_11 environment.
ml purge
ml Miniforge3

ENV_DIR="$SHARED_SCRATCH/$USER/envs/patterns5k_3_11"
ENV_YML="$(dirname "$0")/environment.yml"

if [ ! -d "$ENV_DIR" ]; then
  echo "Creating conda env at $ENV_DIR from $ENV_YML ..."
  conda env create -f "$ENV_YML" -p "$ENV_DIR" --yes
fi
conda activate "$ENV_DIR"

mkdir -p slurm_logs

echo "========================================"
echo "SLURM_JOB_ID   = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "Hostname       = $(hostname)"
echo "PWD            = $(pwd)"
echo "CMD            = $1"
echo "========================================"

srun bash -lc "$1"
