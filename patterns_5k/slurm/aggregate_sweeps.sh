#!/bin/bash
# =====================================================================
# Aggregate results from all three sweeps after jobs finish.
#
# Usage:
#   bash aggregate_sweeps.sh <base_cnn_dir> <hist_cnn_dir> <cache_cnn_dir>
#
# In NOTS, you can use srun to get an interactive shell on a compute node, then run the following:
# ml purge
# ml Miniforge3
# eval "$(conda shell.bash hook)"
# conda activate /scratch/sb272/envs/patterns5k_3_11
# then you can run this shell script
#
# Results now default to $SHARED_SCRATCH/$USER/patterns5k_results/
# Example:
#   bash aggregate_sweeps.sh \
#       $SHARED_SCRATCH/$USER/patterns5k_results/slurm_base_cnn_2026-03-02_14-37-48 \
#       $SHARED_SCRATCH/$USER/patterns5k_results/slurm_hist_cnn_2026-03-02_14-37-49
# =====================================================================

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <sweep_dir_1> [sweep_dir_2] [sweep_dir_3]"
  echo ""
  echo "Aggregates results from finished SLURM sweep directories."
  echo "Each directory should contain trial sub-folders with summary_metrics.json."
  exit 1
fi

# Map sweep dirs to their YAML configs by reading the saved copy
for SWEEP_DIR in "$@"; do
  if [ ! -d "$SWEEP_DIR" ]; then
    echo "ERROR: $SWEEP_DIR does not exist. Skipping."
    continue
  fi

  SWEEP_CONFIG="${SWEEP_DIR}/sweep_config.yaml"
  if [ ! -f "$SWEEP_CONFIG" ]; then
    echo "ERROR: No sweep_config.yaml found in $SWEEP_DIR. Skipping."
    continue
  fi

  echo "========================================"
  echo "Aggregating: ${SWEEP_DIR}"
  echo "========================================"
  python slurm/slurm_sweep.py --sweep-config "$SWEEP_CONFIG" --aggregate "$SWEEP_DIR"
  echo
done

echo "Done. Check sweep_results.csv and sweep_summary.csv in each directory."
