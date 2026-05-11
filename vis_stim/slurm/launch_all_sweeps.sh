#!/bin/bash
# =====================================================================
# Launch all vis_stim HP sweeps on NOTS.
#
# Run from the vis_stim project root.
#
# Usage:
#   bash slurm/launch_all_sweeps.sh                  # default settings
#   bash slurm/launch_all_sweeps.sh --dry-run        # preview only
#   bash slurm/launch_all_sweeps.sh --seeds 5
#   bash slurm/launch_all_sweeps.sh --tasks-per-job 5
# =====================================================================

set -euo pipefail

SEEDS=3
START_SEED=0
DRY_RUN=""
SLURM_TEMPLATE="slurm/nots_cpu.slurm"
MAX_ARRAY=500
CONCURRENT=20
TASKS_PER_JOB=10

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)          DRY_RUN="--dry-run"; shift ;;
    --seeds)            SEEDS="$2"; shift 2 ;;
    --start-seed)       START_SEED="$2"; shift 2 ;;
    --template)         SLURM_TEMPLATE="$2"; shift 2 ;;
    --max-array)        MAX_ARRAY="$2"; shift 2 ;;
    --concurrent)       CONCURRENT="$2"; shift 2 ;;
    --tasks-per-job)    TASKS_PER_JOB="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "========================================"
echo "  vis_stim SLURM HP Sweeps (array mode)"
echo "  Seeds: ${SEEDS} (starting at ${START_SEED})"
echo "  Tasks per job: ${TASKS_PER_JOB}"
echo "  Template: ${SLURM_TEMPLATE}"
echo "  Max array size: ${MAX_ARRAY}"
echo "  Concurrent tasks: ${CONCURRENT}"
echo "  Dry-run: ${DRY_RUN:-no}"
echo "========================================"
echo

mkdir -p slurm/logs

echo ">>> Launching SimpleCausalSpikeCNN sweep..."
python slurm/slurm_sweep.py \
    --sweep-config slurm/slurm_sweep_simple_cnn.yaml \
    --seeds "${SEEDS}" \
    --start-seed "${START_SEED}" \
    --slurm-template "${SLURM_TEMPLATE}" \
    --max-array-size "${MAX_ARRAY}" \
    --concurrent "${CONCURRENT}" \
    --tasks-per-job "${TASKS_PER_JOB}" \
    ${DRY_RUN}

echo

if [[ -n "${SHARED_SCRATCH:-}" ]]; then
    RESULTS_ROOT="${SHARED_SCRATCH}/${USER}/vis_stim_results"
else
    RESULTS_ROOT="results"
fi

echo "========================================"
echo "  Sweep submitted."
echo "  Results:    ${RESULTS_ROOT}/"
echo "  Monitor:    squeue -u \$USER"
echo "  Aggregate:  bash slurm/aggregate_sweeps.sh ${RESULTS_ROOT}/vis_stim_simple_cnn_*"
echo "========================================"
