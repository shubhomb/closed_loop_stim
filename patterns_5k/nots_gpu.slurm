#!/bin/bash
# ===========================================================================
# SLURM job template for NOTS (Rice CRC) — GPU node
#
# Usage:  sbatch nots_gpu.slurm "python slurm_experiment.py --config ..."
#
# Requests a single GPU (V100 or A40), 4 CPU cores, and 16 GB RAM per core.
# Adjust --time, --mem-per-cpu, and --gres as needed for your workload.
# ===========================================================================
#SBATCH --job-name=patterns5k
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/patterns5k_%A_%a.out
#SBATCH --error=slurm_logs/patterns5k_%A_%a.err
#SBATCH --mail-type=FAIL

# ---- Prevent numpy / torch from spawning excess threads ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ---- Load environment ----
# Use Miniforge (conda-forge) to reproduce the local 3_11 environment.
# On first run this creates the env under $SHARED_SCRATCH; subsequent runs
# reuse it.  To force a rebuild, delete $ENV_DIR.
ml purge
ml Miniforge3

ENV_DIR="$SHARED_SCRATCH/$USER/envs/patterns5k_3_11"
ENV_YML="$(dirname "$0")/environment.yml"

if [ ! -d "$ENV_DIR" ]; then
  echo "Creating conda env at $ENV_DIR from $ENV_YML ..."
  conda env create -f "$ENV_YML" -p "$ENV_DIR" --yes
fi
conda activate "$ENV_DIR"

# ---- Create log directory (from submit host pwd) ----
mkdir -p slurm_logs

# ---- Print diagnostics ----
echo "========================================"
echo "SLURM_JOB_ID        = $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "SLURM_NODELIST      = $SLURM_NODELIST"
echo "Hostname             = $(hostname)"
echo "GPU                  = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "PWD                  = $(pwd)"

# ---- Determine the command(s) to run ----
# Supports two modes:
#   1. Array-job mode:  $1 is a manifest file (one command per line).
#                       $2 (optional, default 1) is tasks-per-job (packing).
#                       Lines [TASK_ID*N .. TASK_ID*N+N-1] are run sequentially.
#   2. Single-job mode: $1 is the command string itself.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -f "$1" ]]; then
    MANIFEST="$1"
    TASKS_PER_JOB=${2:-1}
    TOTAL_LINES=$(wc -l < "$MANIFEST")
    FIRST=$((SLURM_ARRAY_TASK_ID * TASKS_PER_JOB + 1))   # sed is 1-indexed
    LAST=$((FIRST + TASKS_PER_JOB - 1))
    echo "MANIFEST             = $MANIFEST"
    echo "TASKS_PER_JOB        = $TASKS_PER_JOB"
    echo "ARRAY TASK           = $SLURM_ARRAY_TASK_ID  →  lines $FIRST-$LAST / $TOTAL_LINES"
    echo "========================================"

    for LINE_NUM in $(seq "$FIRST" "$LAST"); do
        CMD=$(sed -n "${LINE_NUM}p" "$MANIFEST")
        [[ -z "$CMD" ]] && continue   # past end of file
        echo ""
        echo ">>> Sub-task $LINE_NUM / $TOTAL_LINES"
        echo "CMD = $CMD"
        eval "$CMD"
        EXIT_CODE=$?
        echo "<<< Sub-task $LINE_NUM finished (exit $EXIT_CODE)"
    done
else
    CMD="$1"
    echo "CMD                  = $CMD"
    echo "========================================"
    srun bash -lc "$CMD"
fi
