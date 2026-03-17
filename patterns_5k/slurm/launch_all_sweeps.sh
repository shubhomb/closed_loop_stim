#!/bin/bash
# =====================================================================
# Launch all two HP sweeps on NOTS (base CNN, history CNN)
#
# Sweeps are submitted SEQUENTIALLY via SLURM dependencies: the history
# CNN sweep waits for the base CNN sweep to finish, and the cache CNN
# sweep waits for the history CNN sweep.  This avoids flooding the
# scheduler with tasks that sit in "Priority" pending state.
#
# Uses SLURM job arrays with trial packing: multiple experiments are
# run sequentially within each array task.
#
# With default settings (tasks-per-job=50, 5 seeds):
#   Base CNN:  720 trials  →  15 array tasks
#   Hist CNN: 2100 trials  →  42 array tasks
#   Total:                    57 array tasks  (2 sbatch submissions, sequential)
#
# Usage:
#   bash launch_all_sweeps.sh [--dry-run] [--seeds N] [--start-seed S]
#
# Examples:
#   bash launch_all_sweeps.sh                     # submit all jobs (5 seeds each)
#   bash launch_all_sweeps.sh --dry-run            # preview commands only
#   bash launch_all_sweeps.sh --seeds 3            # 3 seeds per config
#   bash launch_all_sweeps.sh --tasks-per-job 5    # fewer trials per SLURM task
#   bash launch_all_sweeps.sh --parallel           # submit all at once (no deps)
# =====================================================================

set -euo pipefail

SEEDS=5
START_SEED=0
DRY_RUN=""
SLURM_TEMPLATE="slurm/nots_cpu.slurm"
MAX_ARRAY=500       # max array tasks per sbatch submission
CONCURRENT=20       # max simultaneously running array tasks
TASKS_PER_JOB=50    # trials packed into each array task (sequential)
PARALLEL=false      # if true, submit all sweeps without dependencies

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
    --parallel)         PARALLEL=true; shift ;;
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
echo "  Sequential: $( [[ "$PARALLEL" == true ]] && echo no || echo yes )"
echo "  Dry-run: ${DRY_RUN:-no}"
echo "========================================"
echo

# Create slurm log directory
mkdir -p slurm/logs

# Helper: extract job IDs from slurm_sweep.py output
# slurm_sweep.py prints "Submitted array job <ID>: ..." for each submission
extract_job_ids() {
    grep -oP 'Submitted array job \K[0-9]+' || true
}

# Build dependency flag from a list of job IDs
make_dep_flag() {
    local ids="$1"
    if [[ -z "$ids" || "$PARALLEL" == true ]]; then
        echo ""
    else
        # afterany = start after ALL listed jobs finish (success or fail)
        echo "--dependency afterany:$(echo "$ids" | paste -sd: -)"
    fi
}

# --- 1. Base CNN (no history) ---
echo ">>> Launching base CNN sweep..."
BASE_OUTPUT=$(python slurm/slurm_sweep.py \
    --sweep-config slurm/slurm_sweep_base_cnn.yaml \
    --seeds "${SEEDS}" \
    --start-seed "${START_SEED}" \
    --slurm-template "${SLURM_TEMPLATE}" \
    --max-array-size "${MAX_ARRAY}" \
    --concurrent "${CONCURRENT}" \
    --tasks-per-job "${TASKS_PER_JOB}" \
    ${DRY_RUN} 2>&1)
echo "$BASE_OUTPUT"
BASE_IDS=$(echo "$BASE_OUTPUT" | extract_job_ids)
echo

# --- 2. History CNN (depends on base CNN) ---
DEP_FLAG=$(make_dep_flag "$BASE_IDS")
echo ">>> Launching history CNN sweep..."
[[ -n "$DEP_FLAG" ]] && echo "    (waiting for base CNN jobs: ${BASE_IDS//$'\n'/, })"
HIST_OUTPUT=$(python slurm/slurm_sweep.py \
    --sweep-config slurm/slurm_sweep_hist_cnn.yaml \
    --seeds "${SEEDS}" \
    --start-seed "${START_SEED}" \
    --slurm-template "${SLURM_TEMPLATE}" \
    --max-array-size "${MAX_ARRAY}" \
    --concurrent "${CONCURRENT}" \
    --tasks-per-job "${TASKS_PER_JOB}" \
    ${DEP_FLAG} \
    ${DRY_RUN} 2>&1)
echo "$HIST_OUTPUT"
HIST_IDS=$(echo "$HIST_OUTPUT" | extract_job_ids)
echo

# Determine results directory (matches slurm_sweep.py default)
if [[ -n "${SHARED_SCRATCH:-}" ]]; then
    RESULTS_ROOT="${SHARED_SCRATCH}/${USER}/patterns5k_results"
else
    RESULTS_ROOT="results"
fi

echo "========================================"
echo "  All sweeps submitted (sequential mode)."
echo "  Base CNN  → Hist CNN"
echo "  Results:    ${RESULTS_ROOT}/"
echo "  Monitor:    squeue -u \$USER"
echo "  Aggregate:  bash slurm/aggregate_sweeps.sh ${RESULTS_ROOT}/slurm_base_cnn_* ${RESULTS_ROOT}/slurm_hist_cnn_*"
echo "========================================"
