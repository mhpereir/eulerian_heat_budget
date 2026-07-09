#!/bin/bash
#PBS -N eulerian_heat_budget_prod
#PBS -J 0-85%5
#PBS -l select=1:ncpus=8:mem=25gb
#PBS -j oe
#PBS -o /dev/null
# PBS -o /home/mhpereir/eulerian_heat_budget/logs/

LOGFILE="/home/mhpereir/eulerian_heat_budget/logs/${PBS_JOBID}_EHB_prod.log"
exec > >(tee -a "${LOGFILE}") 2>&1


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/mhpereir/eulerian_heat_budget}"
SCHEDULER_DIR="${SCHEDULER_DIR:-${PROJECT_ROOT}/schedulers}"
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/scripts}"
SETTINGS_FILE="${PRODUCTION_RUN_CLI_SETTINGS:-${SCHEDULER_DIR}/production_run_cli_settings.sh}"

source "${SETTINGS_FILE}"

mkdir -p "${PRODUCTION_OUTPUT_DIR}"

cd "${SCRIPT_DIR}"

COMMON_RUN_ARGS=()
ehb_build_production_run_budget_args COMMON_RUN_ARGS

initialize_manifest() {
  echo "[info] $(date -Is) initializing production manifest in ${PRODUCTION_OUTPUT_DIR}"
  /usr/bin/time -v python run_budget.py \
    "${COMMON_RUN_ARGS[@]}" \
    --init-production-manifest \
    --production-start-year "${START_YEAR}" \
    --production-end-year "${END_YEAR}"
  echo "[info] $(date -Is) manifest initialization complete"
}

ensure_manifest() {
  local waited=0

  if [[ -f "${MANIFEST_PATH}" ]]; then
    return 0
  fi

  while true; do
    if [[ -f "${MANIFEST_PATH}" ]]; then
      return 0
    fi

    if mkdir "${MANIFEST_LOCK_DIR}" 2>/dev/null; then
      if [[ -f "${MANIFEST_PATH}" ]]; then
        rmdir "${MANIFEST_LOCK_DIR}" || true
        return 0
      fi

      if initialize_manifest; then
        rmdir "${MANIFEST_LOCK_DIR}" || true
        return 0
      fi

      local status=$?
      rmdir "${MANIFEST_LOCK_DIR}" || true
      return "${status}"
    fi

    if (( waited >= MANIFEST_WAIT_SECONDS )); then
      echo "[error] Timed out waiting for production manifest at ${MANIFEST_PATH}" >&2
      return 1
    fi

    echo "[info] $(date -Is) waiting for production manifest at ${MANIFEST_PATH}"
    sleep 5
    waited=$((waited + 5))
  done
}

if [[ "${INIT_MANIFEST_ONLY}" == "1" ]]; then
  ensure_manifest
  exit 0
fi

: "${PBS_ARRAY_INDEX:?PBS_ARRAY_INDEX must be set for yearly production runs}"

YEAR=$(ehb_production_year_for_task "${PBS_ARRAY_INDEX}")
ehb_validate_production_year "${YEAR}"
ehb_build_production_time_window "${YEAR}" TIME_START TIME_END

ensure_manifest

echo "[info] $(date -Is) starting production year ${YEAR} on host $(hostname)"
echo "[info] repo root: ${PROJECT_ROOT}"
echo "[info] settings file: ${SETTINGS_FILE}"
echo "[info] output dir: ${PRODUCTION_OUTPUT_DIR}"
/usr/bin/time -v python run_budget.py \
  "${COMMON_RUN_ARGS[@]}" \
  --time-start "${TIME_START}" \
  --time-end "${TIME_END}"
echo "[info] $(date -Is) finished production year ${YEAR}"
