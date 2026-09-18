#!/usr/bin/env bash
# Submit a Slurm job on CARC from the local machine.
# Usage:
#   ./carc/scripts/submit.sh carc/jobs/bootstrap_env.job
#   ./carc/scripts/submit.sh carc/jobs/smoke.job
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/carc/env.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <job-file-relative-to-repo> [sbatch-args...]"
  exit 1
fi

JOB_REL="$1"
shift || true

if [[ ! -f "${ROOT}/${JOB_REL}" ]]; then
  echo "Job file not found: ${ROOT}/${JOB_REL}"
  exit 1
fi

echo "Syncing before submit..."
"${ROOT}/carc/scripts/sync_to_carc.sh"

echo "Submitting ${JOB_REL} on ${CARC_HOST}..."
ssh -o BatchMode=yes "${CARC_HOST}" \
  "mkdir -p '${CARC_REPO_DIR}/carc/logs' && cd '${CARC_REPO_DIR}' && sbatch $* '${JOB_REL}'"
