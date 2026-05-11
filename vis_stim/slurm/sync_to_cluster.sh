#!/bin/bash
# =====================================================================
# Sync vis_stim project to NOTS.
#
# Usage:
#   bash slurm/sync_to_cluster.sh             # code + data
#   bash slurm/sync_to_cluster.sh --code-only
#   bash slurm/sync_to_cluster.sh --data-only
#   bash slurm/sync_to_cluster.sh --dry-run
#
# Code goes to ~/vis_stim, data to /scratch/<user>/vis_stim_data.
# Patterns_5k is also synced because vis_stim imports from it.
# =====================================================================

set -euo pipefail

CLUSTER_USER="sb272"
CLUSTER_HOST="nots.rice.edu"
CLUSTER_DEST="${CLUSTER_USER}@${CLUSTER_HOST}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_VIS_STIM="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_PATTERNS_5K="$(cd "$LOCAL_VIS_STIM/../patterns_5k" && pwd)"

REMOTE_VIS_STIM="~/vis_stim"
REMOTE_PATTERNS_5K="~/patterns_5k"
REMOTE_DATA="/scratch/${CLUSTER_USER}/vis_stim_data"

DRY_RUN=""
SYNC_CODE=true
SYNC_DATA=true

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)     DRY_RUN="--dry-run"; shift ;;
    --code-only)   SYNC_DATA=false; shift ;;
    --data-only)   SYNC_CODE=false; shift ;;
    --help|-h)     head -20 "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "========================================"
echo "  Sync vis_stim to NOTS"
echo "  Code:   ${LOCAL_VIS_STIM}  ->  ${CLUSTER_DEST}:${REMOTE_VIS_STIM}"
echo "  Deps:   ${LOCAL_PATTERNS_5K}  ->  ${CLUSTER_DEST}:${REMOTE_PATTERNS_5K}"
echo "  Data:   ${CLUSTER_DEST}:${REMOTE_DATA}"
echo "  Dry-run: ${DRY_RUN:-no}"
echo "========================================"
echo

RSYNC_OPTS="-avz --progress ${DRY_RUN}"

if [[ "$SYNC_CODE" == true ]]; then
    echo ">>> Syncing vis_stim code..."
    rsync ${RSYNC_OPTS} \
        --exclude='__pycache__/' \
        --exclude='.DS_Store' \
        --exclude='*.pyc' \
        --exclude='.vscode/' \
        --exclude='data/' \
        --exclude='results/' \
        --exclude='*.ipynb' \
        "${LOCAL_VIS_STIM}/" \
        "${CLUSTER_DEST}:${REMOTE_VIS_STIM}/"

    echo ">>> Syncing patterns_5k code (vis_stim depends on it)..."
    rsync ${RSYNC_OPTS} \
        --exclude='__pycache__/' \
        --exclude='.DS_Store' \
        --exclude='*.pyc' \
        --exclude='data/' \
        --exclude='results/' \
        --exclude='sweep_dirs/' \
        --exclude='rasters/' \
        --exclude='archive/' \
        --exclude='*.ipynb' \
        "${LOCAL_PATTERNS_5K}/" \
        "${CLUSTER_DEST}:${REMOTE_PATTERNS_5K}/"
    echo
fi

if [[ "$SYNC_DATA" == true ]]; then
    echo ">>> Syncing vis_stim data to scratch..."
    ssh "${CLUSTER_DEST}" "mkdir -p ${REMOTE_DATA}"
    rsync ${RSYNC_OPTS} \
        --exclude='.DS_Store' \
        "${LOCAL_VIS_STIM}/data/" \
        "${CLUSTER_DEST}:${REMOTE_DATA}/"
    echo
fi

echo "========================================"
echo "  Sync complete. Edit slurm/slurm_sweep_simple_cnn.yaml's datadir"
echo "  field to point at: ${REMOTE_DATA}"
echo "========================================"
