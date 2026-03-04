#!/bin/bash
# =====================================================================
# Interactive env setup & verification for NOTS.
#
# Run this ONCE on a compute node before submitting any sweep jobs.
#
# NOTS has a known issue: the xalt module puts a stale libcrypto.so
# on LD_LIBRARY_PATH, which breaks Python's ssl module.
#
# Strategy: install ALL packages via conda (conda uses its own libcurl,
# not Python's ssl module).
#
# Usage (from project root on NOTS):
#   srun --partition=commons --account=commons
#        --cpus-per-task=2 --mem=8G --time=00:30:00 --pty bash
#   bash slurm/setup_nots_env.sh
# =====================================================================

set -euo pipefail

# ---------- helper: strip xalt from LD_LIBRARY_PATH ----------
strip_xalt() {
    export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" \
        | tr ':' '\n' | grep -v '/opt/apps/xalt' | paste -sd ':')
    unset LD_PRELOAD 2>/dev/null || true
}

echo "========================================"
echo "  NOTS Environment Setup"
echo "  Host: $(hostname)"
echo "========================================"

# 1. Load Miniforge and defuse xalt
ml purge
ml Miniforge3
strip_xalt

MINIFORGE_LIB=$(python3 -c "import sys; print(sys.prefix)")/lib
export LD_LIBRARY_PATH="$MINIFORGE_LIB:${LD_LIBRARY_PATH:-}"

# Using default solver (libmamba). 
export CONDA_PKGS_DIRS="$SHARED_SCRATCH/$USER/conda_pkgs"
mkdir -p "$CONDA_PKGS_DIRS"
eval "$(conda shell.bash hook)"
strip_xalt                       # conda shell hook may re-source profiles
export LD_LIBRARY_PATH="$MINIFORGE_LIB:${LD_LIBRARY_PATH:-}"

ENV_DIR="$SHARED_SCRATCH/$USER/envs/patterns5k_3_11"

echo "ENV_DIR         = $ENV_DIR"
echo "CONDA_PKGS_DIRS = $CONDA_PKGS_DIRS"
echo ""

# 2. Create env if it doesn't exist
if [ -d "$ENV_DIR" ]; then
    echo ">>> Env already exists. To rebuild, rename it first:"
    echo "    mv $ENV_DIR ${ENV_DIR}_old"
    echo ">>> Proceeding to activate and verify..."
else
    # ------------------------------------------------------------------
    # 3. Install Python + ALL scientific packages via conda
    #    (conda uses libcurl for downloads — immune to Python SSL issues)
    # ------------------------------------------------------------------
    echo ">>> Creating env with conda (including PyTorch)..."
    conda create -p "$ENV_DIR" \
        -c pytorch -c nvidia -c conda-forge --yes -q \
        python=3.11 openssl pip setuptools wheel \
        pytorch torchvision pytorch-cuda=12.4 \
        numpy=2.3.3 scipy pandas=2.3.3 scikit-learn matplotlib h5py \
        networkx sympy tqdm pyyaml pillow requests joblib psutil \
        einops scikit-image umap-learn \
        filelock typing_extensions jinja2 fsspec

    # 4. Activate and fix LD_LIBRARY_PATH
    conda activate "$ENV_DIR"
    strip_xalt
    export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

    echo "Python: $(which python)"
    echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    echo ">>> PyTorch installation and setup complete!"
fi