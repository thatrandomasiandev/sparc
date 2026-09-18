#!/usr/bin/env bash
# Bootstrap the research conda env on a CARC compute node (salloc/sbatch).
# Prefer: sbatch carc/jobs/bootstrap_env.job
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/carc/env.sh"

if [[ -z "${CARC_NETID}" ]]; then
  export CARC_NETID="${USER}"
  # shellcheck source=/dev/null
  source "${ROOT}/carc/env.sh"
fi

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi

module purge
module load gcc/13.3.0 2>/dev/null || module load gcc/11.3.0 || true
module load cuda/12.4.0 2>/dev/null || module load cuda/12.6.3 || true

mkdir -p "$(dirname "${CARC_CONDA_DIR}")" "${CARC_SCRATCH_DIR}/experiments/runs"

if [[ ! -x "${CARC_CONDA_DIR}/bin/conda" ]]; then
  echo "Installing Miniconda to ${CARC_CONDA_DIR}"
  cd /tmp
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p "${CARC_CONDA_DIR}"
fi

# shellcheck source=/dev/null
source "${CARC_CONDA_DIR}/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^${CARC_ENV_NAME}[[:space:]]"; then
  conda create -y -n "${CARC_ENV_NAME}" python=3.11
fi
conda activate "${CARC_ENV_NAME}"

pip install -U pip setuptools wheel

cd "${ROOT}"
pip install -e ".[dev,plot]"

python - <<'PY'
import research
print("research", research.__version__)
PY

echo "Bootstrap complete: conda activate ${CARC_ENV_NAME}"
echo "Add domain deps (torch / gym / etc.) once the topic locks."
