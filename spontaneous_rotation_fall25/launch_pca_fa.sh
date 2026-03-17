#!/bin/bash
# ===========================================================================
# launch_pca_fa.sh  –  Run PCA/FA comparison locally or via SLURM (Rice NOTS)
#
# Usage:
#   bash launch_pca_fa.sh                  # both methods, local
#   bash launch_pca_fa.sh --slurm          # both methods, separate SLURM jobs
#   bash launch_pca_fa.sh --method PCA     # only PCA, local
#   bash launch_pca_fa.sh --method FA --slurm
#
# Optional env overrides:
#   OUTPUT_DIR=/my/path bash launch_pca_fa.sh --slurm
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/compare_pca_fa.py"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
METHOD="both"
USE_SLURM=false
SLURM_TEMPLATE="$SCRIPT_DIR/slurm/run_pca_fa.slurm"
EXTRA_ARGS=""

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm)        USE_SLURM=true; shift ;;
        --method)       METHOD="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        *)              EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# Normalise method aliases
[[ "$METHOD" == "FA" ]] && METHOD="FactorAnalysis"

mkdir -p "$OUTPUT_DIR"

# ---- Build method list ----
if [[ "$METHOD" == "both" ]]; then
    METHODS=("PCA" "FactorAnalysis")
else
    METHODS=("$METHOD")
fi

if $USE_SLURM; then
    # ---- SLURM mode: one job per method ----
    MANIFEST="$SCRIPT_DIR/slurm/pca_fa_manifest.txt"
    : > "$MANIFEST"  # truncate

    for M in "${METHODS[@]}"; do
        METHOD_OUT="$OUTPUT_DIR/$M"
        echo "python $PYTHON_SCRIPT --method $M --output-dir $METHOD_OUT $EXTRA_ARGS" >> "$MANIFEST"
    done

    N_TASKS=$(wc -l < "$MANIFEST")
    LAST_IDX=$((N_TASKS - 1))

    echo "Manifest written: $MANIFEST ($N_TASKS tasks)"
    echo "Submitting SLURM array job (tasks 0-$LAST_IDX)..."

    sbatch --array="0-$LAST_IDX" "$SLURM_TEMPLATE" "$MANIFEST" 1

    echo "Submitted. Monitor with: squeue -u \$USER"
    echo "Results will be saved under: $OUTPUT_DIR/{PCA,FactorAnalysis}/"

else
    # ---- Local mode ----
    echo "Running locally (set OUTPUT_DIR or pass --output-dir to override)."
    for M in "${METHODS[@]}"; do
        METHOD_OUT="$OUTPUT_DIR/$M"
        echo ""
        echo ">>> Starting $M ..."
        python "$PYTHON_SCRIPT" --method "$M" --output-dir "$METHOD_OUT" $EXTRA_ARGS
        echo "<<< $M done. Figure in $METHOD_OUT/"
    done
    echo ""
    echo "All done. Results in $OUTPUT_DIR/"
fi
