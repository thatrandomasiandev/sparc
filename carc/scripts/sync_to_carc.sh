#!/usr/bin/env bash
# Sync this repo to CARC Discovery (key auth).
# Usage (local):
#   ./carc/scripts/sync_to_carc.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/carc/env.sh"

REMOTE_HOST="${CARC_HOST}"
REMOTE_DIR="${CARC_REPO_DIR}"

echo "Syncing ${ROOT} -> ${REMOTE_HOST}:${REMOTE_DIR}"
ssh -o BatchMode=yes "${REMOTE_HOST}" "mkdir -p '${REMOTE_DIR}' '${CARC_SCRATCH_DIR}/experiments/runs' '${REMOTE_DIR}/carc/logs'"

rsync -avz \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '*.ckpt' \
  --exclude 'carc/logs/*.out' \
  --exclude 'carc/logs/*.err' \
  --exclude 'experiments/runs/**' \
  --include 'experiments/runs/.gitkeep' \
  "${ROOT}/" \
  "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Done. On Discovery:"
echo "  cd ${REMOTE_DIR}"
echo "  sbatch carc/jobs/bootstrap_env.job   # first time only"
echo "  sbatch carc/jobs/smoke.job"
