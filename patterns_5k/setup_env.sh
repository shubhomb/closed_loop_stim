#!/bin/bash
# =====================================================================
# One-time environment setup for NOTS.
#
# Run this ONCE (e.g. in an interactive session) before submitting
# sweep jobs.  It creates the conda environment from environment.yml
# so that individual SLURM jobs don't all race to create it.
#
# Usage:
#   srun --pty --partition=commons --ntasks=1 --mem=8G --time=00:30:00 $SHELL
#   bash setup_env.sh
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_YML="${SCRIPT_DIR}/environment.yml"
ENV_DIR="${SHARED_SCRATCH}/${USER}/envs/patterns5k_3_11"

if [ ! -f "$ENV_YML" ]; then
  echo "ERROR: environment.yml not found at $ENV_YML"
  exit 1
fi

echo "Loading Miniforge module..."
ml purge
ml Miniforge3

if [ -d "$ENV_DIR" ]; then
  echo "Environment already exists at $ENV_DIR"
  echo "To rebuild, remove it first:  rm -rf $ENV_DIR"
  echo "Updating existing environment..."
  conda env update -f "$ENV_YML" -p "$ENV_DIR" --prune
else
  echo "Creating environment at $ENV_DIR ..."
  conda env create -f "$ENV_YML" -p "$ENV_DIR" --yes
fi

echo ""
echo "Done. Activate with:  conda activate $ENV_DIR"
echo "Verify with:  python -c 'import torch; print(torch.__version__)'"
