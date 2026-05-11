#!/bin/bash
# =====================================================================
# Local workstation env setup for vis_stim sweeps.
#
# Uses python -m venv + pip (much faster than the conda solver, and the
# slurm env's conda recipe pulls in deps we don't need locally).
#
# Env lives in your user space so it does not affect other users:
#   /home/user/users/<owner>/envs/vis_stim_local_venv
#
# Usage:
#   bash local/setup_local_env.sh
#
# After it finishes:
#   source /home/user/users/<owner>/envs/vis_stim_local_venv/bin/activate
#   python local/local_sweep.py --sweep-config local/local_sweep_test.yaml
# =====================================================================

set -euo pipefail

OWNER="${OWNER:-shubhom}"
ENV_DIR="${ENV_DIR:-/home/user/users/${OWNER}/envs/vis_stim_local_venv}"

# Pick a python that has venv+ensurepip. The system /usr/bin/python3 on
# this workstation is missing python3-venv, so prefer miniconda's base
# python (which has venv built in).
PY_BIN="${PY_BIN:-}"
if [ -z "$PY_BIN" ]; then
    for cand in /home/user/miniconda3/bin/python python3.11 python3.12 python3.10 python3; do
        if [ -x "$cand" ] || command -v "$cand" >/dev/null 2>&1; then PY_BIN="$cand"; break; fi
    done
fi
if [ -z "$PY_BIN" ]; then
    echo "ERROR: no python3 found" >&2; exit 1
fi

echo "========================================"
echo "  vis_stim local venv setup"
echo "  Host:    $(hostname)"
echo "  Env dir: $ENV_DIR"
echo "  Python:  $(${PY_BIN} -V)  ($(command -v ${PY_BIN}))"
echo "========================================"

mkdir -p "$(dirname "$ENV_DIR")"

if [ -d "$ENV_DIR" ]; then
    echo ">>> Venv already exists. To rebuild: rm -rf $ENV_DIR"
else
    echo ">>> Creating venv ..."
    # --without-pip + bootstrap below avoids an ensurepip failure that
    # can hit when the base python's stdlib isn't fully on sys.path.
    "$PY_BIN" -m venv --without-pip "$ENV_DIR"
fi

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

if ! command -v pip >/dev/null 2>&1; then
    echo ">>> Bootstrapping pip ..."
    curl -fsSL https://bootstrap.pypa.io/get-pip.py | python -
fi
python -m pip install --upgrade pip wheel setuptools

# PyTorch with CUDA 12.6 wheels (driver here reports CUDA 12.8 runtime,
# which is forward-compatible with 12.6 binaries).
echo ">>> Installing PyTorch (CUDA 12.6) ..."
pip install --index-url https://download.pytorch.org/whl/cu126 \
    torch torchvision

echo ">>> Installing scientific deps ..."
pip install \
    "numpy<2.4" "pandas>=2.0" \
    scipy scikit-learn matplotlib pillow \
    tqdm pyyaml

echo ""
echo ">>> Verifying install"
python - <<'PY'
import sys, importlib
print(f"Python  : {sys.version.split()[0]}")
print(f"Prefix  : {sys.prefix}")
import torch
print(f"torch   : {torch.__version__}")
print(f"cuda OK : {torch.cuda.is_available()}  ({torch.cuda.device_count()} devices)")
for m in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "PIL", "tqdm", "yaml"):
    importlib.import_module(m)
print("All deps import OK.")
PY

echo ""
echo "========================================"
echo "Activate this venv in future shells with:"
echo "  source $ENV_DIR/bin/activate"
echo "========================================"
