#!/bin/bash
# =====================================================================
# Quick test: launch the 2-trial test sweep to verify the pipeline
#
# Usage (from patterns_5k/):
#   bash slurm/test_launch_sweeps.sh              # submit for real
#   bash slurm/test_launch_sweeps.sh --dry-run     # preview only
# =====================================================================

set -euo pipefail

DRY_RUN=""
SLURM_TEMPLATE="slurm/nots_cpu.slurm"

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)    DRY_RUN="--dry-run"; shift ;;
    --template)   SLURM_TEMPLATE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "========================================"
echo "  Pipeline test sweep (2 trials, 1 seed)"
echo "  Template: ${SLURM_TEMPLATE}"
echo "  Dry-run:  ${DRY_RUN:-no}"
echo "========================================"
echo

mkdir -p slurm/logs

# Submit the test sweep: 2 configs × 1 seed = 2 trials, packed into 1 task
python slurm/slurm_sweep.py \
    --sweep-config slurm/slurm_sweep_test.yaml \
    --seeds 1 \
    --start-seed 0 \
    --slurm-template "${SLURM_TEMPLATE}" \
    --tasks-per-job 2 \
    --max-array-size 10 \
    --concurrent 2 \
    ${DRY_RUN}

echo
echo "========================================"
echo "  Test sweep submitted."
echo "  Monitor:    squeue -u \$USER"
echo "  Aggregate:  python slurm/slurm_sweep.py --sweep-config slurm/slurm_sweep_test.yaml --aggregate \$SHARED_SCRATCH/\$USER/patterns5k_results/slurm_test_*"
echo "========================================"
