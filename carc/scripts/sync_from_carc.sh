#!/usr/bin/env bash
# Pull experiment run artifacts from CARC back to this machine.
# Usage (local):
#   ./carc/scripts/sync_from_carc.sh
#   ./carc/scripts/sync_from_carc.sh experiments/runs/carc_smoke
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/carc/env.sh"

REMOTE_HOST="${CARC_HOST}"
REL_PATH="${1:-experiments/runs}"
REMOTE_PATH="${CARC_REPO_DIR}/${REL_PATH}"
LOCAL_PATH="${ROOT}/${REL_PATH}"

mkdir -p "${LOCAL_PATH}"
echo "Pulling ${REMOTE_HOST}:${REMOTE_PATH}/ -> ${LOCAL_PATH}/"

# Prefer repo copy; fall back to scratch if present.
rsync -avz \
  --exclude '__pycache__/' \
  "${REMOTE_HOST}:${REMOTE_PATH}/" \
  "${LOCAL_PATH}/" || true

if ssh -o BatchMode=yes "${REMOTE_HOST}" "test -d '${CARC_SCRATCH_DIR}/${REL_PATH}'"; then
  echo "Also pulling scratch: ${CARC_SCRATCH_DIR}/${REL_PATH}"
  rsync -avz \
    "${REMOTE_HOST}:${CARC_SCRATCH_DIR}/${REL_PATH}/" \
    "${LOCAL_PATH}/"
fi

echo "Done."
