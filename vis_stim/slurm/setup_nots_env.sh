#!/bin/bash
# =====================================================================
# Verify the shared patterns5k_3_11 conda env satisfies vis_stim deps.
#
# vis_stim re-uses the env created by patterns_5k/slurm/setup_nots_env.sh.
# If you have not run that yet, do so first:
#
#   srun --partition=commons --account=commons \
#        --cpus-per-task=2 --mem=8G --time=00:30:00 --pty bash
#   bash ../patterns_5k/slurm/setup_nots_env.sh
#
# Then run THIS script to install any vis_stim-only extras (plotly).
# =====================================================================

set -euo pipefail

strip_xalt() {
    export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" \
        | tr ':' '\n' | grep -v '/opt/apps/xalt' | paste -sd ':')
    unset LD_PRELOAD 2>/dev/null || true
}

echo "========================================"
echo "  vis_stim env verification"
echo "  Host: $(hostname)"
echo "========================================"

ml purge
ml Miniforge3
strip_xalt

MINIFORGE_LIB=$(python3 -c "import sys; print(sys.prefix)")/lib
export LD_LIBRARY_PATH="$MINIFORGE_LIB:${LD_LIBRARY_PATH:-}"
export CONDA_PKGS_DIRS="$SHARED_SCRATCH/$USER/conda_pkgs"
mkdir -p "$CONDA_PKGS_DIRS"

eval "$(conda shell.bash hook)"
strip_xalt
export LD_LIBRARY_PATH="$MINIFORGE_LIB:${LD_LIBRARY_PATH:-}"

ENV_DIR="$SHARED_SCRATCH/$USER/envs/patterns5k_3_11"

if [ ! -d "$ENV_DIR" ]; then
    echo "ERROR: Shared env not found at $ENV_DIR"
    echo "Run patterns_5k/slurm/setup_nots_env.sh first."
    exit 1
fi

conda activate "$ENV_DIR"
strip_xalt
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

echo "Python: $(which python)"

# vis_stim only adds plotly on top of the patterns5k stack (used by run_greedy.py;
# run_experiment.py itself does not need it but this keeps the env consistent).
python -c "import plotly" 2>/dev/null || conda install -p "$ENV_DIR" -c conda-forge -y plotly

# Quick import check
python - <<'PY'
import sys
print("Python:", sys.version)
import numpy, scipy, pandas, sklearn, torch, matplotlib, yaml
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
PY

echo ">>> vis_stim env ready."
