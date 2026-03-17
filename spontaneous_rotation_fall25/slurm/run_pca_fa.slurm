#!/bin/bash
# ===========================================================================
# SLURM job script for compare_pca_fa.py on Rice NOTS (CPU node)
#
# Usage (single job):
#   sbatch slurm/run_pca_fa.slurm "python compare_pca_fa.py --method PCA"
#   sbatch slurm/run_pca_fa.slurm "python compare_pca_fa.py --method FactorAnalysis"
#   sbatch slurm/run_pca_fa.slurm "python compare_pca_fa.py --method both"
#
# Usage (array job via manifest – see launch_pca_fa.sh):
#   sbatch --array=0-1 slurm/run_pca_fa.slurm slurm/pca_fa_manifest.txt
# ===========================================================================
#SBATCH --job-name=pca_fa_compare
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm/logs/pca_fa_%A_%a.out
#SBATCH --error=slurm/logs/pca_fa_%A_%a.err
#SBATCH --mail-user=sb272@rice.edu
#SBATCH --mail-type=END,FAIL

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ---- Load environment ----
ml purge
ml Miniforge3

export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/opt/apps/xalt' | paste -sd ':')
unset LD_PRELOAD 2>/dev/null || true
MINIFORGE_LIB=$(python3 -c "import sys; print(sys.prefix)")/lib
export LD_LIBRARY_PATH="$MINIFORGE_LIB:${LD_LIBRARY_PATH:-}"
export CONDA_SOLVER=classic
export CONDA_PKGS_DIRS="$SHARED_SCRATCH/$USER/conda_pkgs"

eval "$(conda shell.bash hook)"

export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/opt/apps/xalt' | paste -sd ':')

ENV_DIR="$SHARED_SCRATCH/$USER/envs/patterns5k_3_11"
if [ ! -d "$ENV_DIR" ]; then
  echo "ERROR: Conda env not found at $ENV_DIR"
  echo "Run 'bash ../patterns_5k/slurm/setup_nots_env.sh' on a compute node first."
  exit 1
fi
conda activate "$ENV_DIR"

export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/opt/apps/xalt' | paste -sd ':')
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

mkdir -p slurm/logs

echo "========================================"
echo "SLURM_JOB_ID        = $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "SLURM_NODELIST      = $SLURM_NODELIST"
echo "Hostname             = $(hostname)"
echo "PWD                  = $(pwd)"

# ---- Dispatch: array-job manifest OR single command ----
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -f "$1" ]]; then
    MANIFEST="$1"
    TASKS_PER_JOB=${2:-1}
    FIRST=$((SLURM_ARRAY_TASK_ID * TASKS_PER_JOB + 1))
    LAST=$((FIRST + TASKS_PER_JOB - 1))
    TOTAL_LINES=$(wc -l < "$MANIFEST")
    echo "MANIFEST   = $MANIFEST"
    echo "TASK range = lines $FIRST-$LAST / $TOTAL_LINES"
    echo "========================================"
    for LINE_NUM in $(seq "$FIRST" "$LAST"); do
        CMD=$(sed -n "${LINE_NUM}p" "$MANIFEST")
        [[ -z "$CMD" ]] && continue
        echo ">>> Sub-task $LINE_NUM: $CMD"
        eval "$CMD"
        echo "<<< exit $?"
    done
else
    CMD="$1"
    echo "CMD = $CMD"
    echo "========================================"
    srun bash -lc "$CMD"
fi
