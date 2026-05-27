#!/bin/bash
#PBS -N eulerian_head_budget
#PBS -l select=1:ncpus=12:mem=32gb
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail

resolve_repo_root() {
  if [[ -n "${PROJECT_ROOT:-}" ]]; then
    cd "${PROJECT_ROOT}" && pwd
    return
  fi

  local submit_dir="${PBS_O_WORKDIR:-$PWD}"
  if [[ -d "${submit_dir}/scripts" ]]; then
    cd "${submit_dir}" && pwd
    return
  fi
  if [[ -d "${submit_dir}/../scripts" ]]; then
    cd "${submit_dir}/.." && pwd
    return
  fi

  local scheduler_dir
  scheduler_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  if [[ -d "${scheduler_dir}/../scripts" ]]; then
    cd "${scheduler_dir}/.." && pwd
    return
  fi

  echo "[error] Unable to resolve repository root. Set PROJECT_ROOT explicitly." >&2
  exit 1
}

REPO_ROOT=$(resolve_repo_root)
SCRIPT_DIR="${REPO_ROOT}/scripts"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

mkdir -p "${LOG_DIR}"

LOGFILE="${LOG_DIR}/${PBS_JOBID:-manual}_EHB_6hr_single.log"
exec > >(tee -a "${LOGFILE}") 2>&1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate dev_env

TIME_START="1941-06-01T00:00:00"
TIME_END="1941-09-01T00:00:00"
REGION="${REGION:-pnw_bartusek}"
ENABLE_BENCHMARK_VARIABLES="${ENABLE_BENCHMARK_VARIABLES:-0}"

cd "${SCRIPT_DIR}"

RUN_ARGS=(
  --data-source arco_era5
  --region "${REGION}"
  --time-start "${TIME_START}"
  --time-end "${TIME_END}"
  --diagnostic-plots
  --constant-temperature-test
  --six-hourly-phases
)

if [[ "${ENABLE_BENCHMARK_VARIABLES}" == "1" ]]; then
  RUN_ARGS+=(--include-benchmark-variables)
fi

echo "[info] $(date -Is) starting eulerian heat budget calculation on host $(hostname)"
/usr/bin/time -v python run_budget.py \
  "${RUN_ARGS[@]}"
echo "[info] $(date -Is) done"
