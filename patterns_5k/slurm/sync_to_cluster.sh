#!/bin/bash
# =====================================================================
# Sync local project files to the NOTS cluster
#
# Transfers code, SLURM scripts, and data to the cluster in one command.
# Uses rsync for efficient incremental transfers.
#
# Usage:
#   bash sync_to_cluster.sh                # sync code + slurm + data
#   bash sync_to_cluster.sh --code-only    # skip data (faster)
#   bash sync_to_cluster.sh --dry-run      # preview what would transfer
#   bash sync_to_cluster.sh --data-only    # only sync data to scratch
#
# Prerequisites:
#   - SSH key or password access to nots.rice.edu
#   - (Optional) Add to ~/.ssh/config for convenience:
#       Host nots
#           HostName nots.rice.edu
#           User sb272
# =====================================================================

set -euo pipefail

# ---- Configuration (edit these) ----
CLUSTER_USER="sb272"
CLUSTER_HOST="nots.rice.edu"
CLUSTER_DEST="${CLUSTER_USER}@${CLUSTER_HOST}"

# Local project root (where this script lives: patterns_5k/slurm/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Remote paths
REMOTE_CODE="~/patterns_5k"                                    # home dir (code is small)
REMOTE_DATA="/scratch/${CLUSTER_USER}/patterns5k_data"         # scratch (data is large)

# ---- Parse flags ----
DRY_RUN=""
SYNC_CODE=true
SYNC_DATA=true

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)     DRY_RUN="--dry-run"; shift ;;
    --code-only)   SYNC_DATA=false; shift ;;
    --data-only)   SYNC_CODE=false; shift ;;
    --help|-h)
      head -20 "$0" | grep '^#' | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "========================================"
echo "  Sync to NOTS cluster"
echo "  Local:  ${LOCAL_PROJECT}"
echo "  Remote: ${CLUSTER_DEST}:${REMOTE_CODE}"
echo "  Data:   ${CLUSTER_DEST}:${REMOTE_DATA}"
echo "  Dry-run: ${DRY_RUN:-no}"
echo "========================================"
echo

# Common rsync flags
RSYNC_OPTS="-avz --progress ${DRY_RUN}"

# ---- Sync code + slurm scripts ----
if [[ "$SYNC_CODE" == true ]]; then
    echo ">>> Syncing code & SLURM scripts..."
    rsync ${RSYNC_OPTS} \
        --exclude='__pycache__/' \
        --exclude='.DS_Store' \
        --exclude='*.pyc' \
        --exclude='.vscode/' \
        --exclude='data/' \
        --exclude='results/' \
        --exclude='sweep_dirs/' \
        --exclude='rasters/' \
        --exclude='archive/' \
        --exclude='*.ipynb' \
        --exclude='hp_tuning_results/' \
        "${LOCAL_PROJECT}/" \
        "${CLUSTER_DEST}:${REMOTE_CODE}/"
    echo "  Done."
    echo
fi

# ---- Sync data to scratch ----
if [[ "$SYNC_DATA" == true ]]; then
    echo ">>> Syncing data to scratch..."

    # Create remote data directory
    ssh "${CLUSTER_DEST}" "mkdir -p ${REMOTE_DATA}"

    rsync ${RSYNC_OPTS} \
        --exclude='.DS_Store' \
        "${LOCAL_PROJECT}/data/" \
        "${CLUSTER_DEST}:${REMOTE_DATA}/"
    echo "  Done."
    echo
fi

echo "========================================"
echo "  Sync complete!"
if [[ "$SYNC_CODE" == true ]]; then
    echo "  Code:  ${CLUSTER_DEST}:${REMOTE_CODE}"
fi
if [[ "$SYNC_DATA" == true ]]; then
    echo "  Data:  ${CLUSTER_DEST}:${REMOTE_DATA}"
fi
echo "========================================"
