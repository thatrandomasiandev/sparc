#!/usr/bin/env bash
# Check queue / recent research jobs from the local machine.
# Usage: ./carc/scripts/status.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/carc/env.sh"

ssh -o BatchMode=yes "${CARC_HOST}" bash -s <<EOF
set -euo pipefail
echo "=== squeue (${CARC_NETID}) ==="
squeue -u "${CARC_NETID}" -o '%.18i %.9P %.30j %.8T %.10M %.9l %R' || true
echo
echo "=== recent logs ==="
ls -lt "${CARC_REPO_DIR}/carc/logs" 2>/dev/null | head -20 || true
EOF
