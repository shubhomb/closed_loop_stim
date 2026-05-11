#!/bin/bash
# =====================================================================
# Aggregate results from one or more vis_stim sweep directories.
#
# Usage (from vis_stim root):
#   bash slurm/aggregate_sweeps.sh <sweep_dir_1> [<sweep_dir_2> ...]
#
# On NOTS, results live at:
#   $SHARED_SCRATCH/$USER/vis_stim_results/vis_stim_simple_cnn_<timestamp>
# =====================================================================

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <sweep_dir_1> [<sweep_dir_2> ...]"
  exit 1
fi

for SWEEP_DIR in "$@"; do
  if [ ! -d "$SWEEP_DIR" ]; then
    echo "ERROR: $SWEEP_DIR does not exist. Skipping."
    continue
  fi

  SWEEP_CONFIG="${SWEEP_DIR}/sweep_config.yaml"
  if [ ! -f "$SWEEP_CONFIG" ]; then
    echo "ERROR: No sweep_config.yaml in $SWEEP_DIR. Skipping."
    continue
  fi

  echo "========================================"
  echo "Aggregating: ${SWEEP_DIR}"
  echo "========================================"
  python slurm/slurm_sweep.py --sweep-config "$SWEEP_CONFIG" --aggregate "$SWEEP_DIR"
  echo
done

echo "Done. See sweep_results.csv and sweep_summary.csv in each directory."
