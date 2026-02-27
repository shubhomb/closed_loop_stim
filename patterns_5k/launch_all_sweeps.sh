#!/bin/bash
# =====================================================================
# Launch all three HP sweeps on NOTS (base CNN, history CNN, cache CNN)
#
# Uses SLURM job arrays with trial packing: multiple experiments are
# run sequentially within each array task, keeping total array-task
# count under the QOS max-submit-per-user limit.
#
# With default settings (tasks-per-job=50, 5 seeds):
#   Base CNN:  720 trials  →  15 array tasks
#   Hist CNN: 2100 trials  →  42 array tasks
#   Cache CNN:4480 trials  →  90 array tasks
#   Total:                    147 array tasks  (3 sbatch submissions)
#
# Usage:
#   bash launch_all_sweeps.sh [--dry-run] [--seeds N] [--start-seed S]
#
# Examples:
#   bash launch_all_sweeps.sh                  # submit all jobs (5 seeds each)
#   bash launch_all_sweeps.sh --dry-run        # preview commands only
#   bash launch_all_sweeps.sh --seeds 3        # 3 seeds per config
#   bash launch_all_sweeps.sh --tasks-per-job 5  # fewer trials per SLURM task
# =====================================================================

set -euo pipefail

SEEDS=5
START_SEED=0
DRY_RUN=""
SLURM_TEMPLATE="nots_gpu.slurm"
MAX_ARRAY=500       # max array tasks per sbatch submission
CONCURRENT=50       # max simultaneously running array tasks
TASKS_PER_JOB=50    # trials packed into each array task (sequential)

# Parse CLI flags
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
echo "  Patterns-5k SLURM HP Sweeps (array mode)"
echo "  Seeds: ${SEEDS} (starting at ${START_SEED})"
echo "  Tasks per job: ${TASKS_PER_JOB}"
echo "  Template: ${SLURM_TEMPLATE}"
echo "  Max array size: ${MAX_ARRAY}"
echo "  Concurrent tasks: ${CONCURRENT}"
echo "  Dry-run: ${DRY_RUN:-no}"
echo "========================================"
echo

# Create slurm_logs directory
mkdir -p slurm_logs

# --- 1. Base CNN (no history) ---
echo ">>> Launching base CNN sweep..."
python slurm_sweep.py \
    --sweep-config slurm_sweep_base_cnn.yaml \
    --seeds "${SEEDS}" \
    --start-seed "${START_SEED}" \
    --slurm-template "${SLURM_TEMPLATE}" \
    --max-array-size "${MAX_ARRAY}" \
    --concurrent "${CONCURRENT}" \
    --tasks-per-job "${TASKS_PER_JOB}" \
    ${DRY_RUN}
echo

# --- 2. History CNN ---
echo ">>> Launching history CNN sweep..."
python slurm_sweep.py \
    --sweep-config slurm_sweep_hist_cnn.yaml \
    --seeds "${SEEDS}" \
    --start-seed "${START_SEED}" \
    --slurm-template "${SLURM_TEMPLATE}" \
    --max-array-size "${MAX_ARRAY}" \
    --concurrent "${CONCURRENT}" \
    --tasks-per-job "${TASKS_PER_JOB}" \
    ${DRY_RUN}
echo

# --- 3. Cache CNN ---
echo ">>> Launching cache CNN sweep..."
python slurm_sweep.py \
    --sweep-config slurm_sweep_cache_cnn.yaml \
    --seeds "${SEEDS}" \
    --start-seed "${START_SEED}" \
    --slurm-template "${SLURM_TEMPLATE}" \
    --max-array-size "${MAX_ARRAY}" \
    --concurrent "${CONCURRENT}" \
    --tasks-per-job "${TASKS_PER_JOB}" \
    ${DRY_RUN}
echo

echo "========================================"
echo "  All sweeps submitted."
echo "  Monitor:    squeue -u \$USER"
echo "  Aggregate:  python slurm_sweep.py --sweep-config <yaml> --aggregate results/<sweep_dir>"
echo "========================================"
