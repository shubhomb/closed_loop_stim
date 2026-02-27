#!/bin/bash
# =====================================================================
# Aggregate results from all three sweeps after jobs finish.
#
# Usage:
#   bash aggregate_sweeps.sh <base_cnn_dir> <hist_cnn_dir> <cache_cnn_dir>
#
# Example:
#   bash aggregate_sweeps.sh \
#       results/slurm_base_cnn_2026-02-26_10-00-00 \
#       results/slurm_hist_cnn_2026-02-26_10-00-05 \
#       results/slurm_cache_cnn_2026-02-26_10-00-10
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
  python slurm_sweep.py --sweep-config "$SWEEP_CONFIG" --aggregate "$SWEEP_DIR"
  echo
done

echo "Done. Check sweep_results.csv and sweep_summary.csv in each directory."
