#!/bin/bash
#PBS -N eulerian_heat_budget_prod
#PBS -J 0-9
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -j oe
#PBS -o /home/mhpereir/eulerian_heat_budget/logs/

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail

START_YEAR=1940
END_YEAR=1949
DATA_SOURCE="${DATA_SOURCE:-arco_era5}"
PRODUCTION_OUTPUT_DIR="${PRODUCTION_OUTPUT_DIR:-/home/mhpereir/eulerian_heat_budget/results/production/pnw_1940_1949}"
INIT_MANIFEST_ONLY="${INIT_MANIFEST_ONLY:-0}"
ENABLE_DIAGNOSTIC_PLOTS="${ENABLE_DIAGNOSTIC_PLOTS:-1}"
ENABLE_CONSTANT_TEMPERATURE_TEST="${ENABLE_CONSTANT_TEMPERATURE_TEST:-0}"
RUN_START_MONTH_DAY="${RUN_START_MONTH_DAY:-05-01}"
RUN_END_MONTH_DAY="${RUN_END_MONTH_DAY:-09-30}"
MANIFEST_PATH="${PRODUCTION_OUTPUT_DIR}/production_run.json"
MANIFEST_LOCK_DIR="${PRODUCTION_OUTPUT_DIR}/.manifest_init.lock"
MANIFEST_WAIT_SECONDS="${MANIFEST_WAIT_SECONDS:-300}"

mkdir -p "${PRODUCTION_OUTPUT_DIR}"

cd /home/mhpereir/eulerian_heat_budget/scripts

COMMON_RUN_ARGS=(
  --data-source "${DATA_SOURCE}"
  --production-output-dir "${PRODUCTION_OUTPUT_DIR}"
)

if [[ "${ENABLE_DIAGNOSTIC_PLOTS}" == "1" ]]; then
  COMMON_RUN_ARGS+=(--diagnostic-plots)
else
  COMMON_RUN_ARGS+=(--no-diagnostic-plots)
fi

if [[ "${ENABLE_CONSTANT_TEMPERATURE_TEST}" == "1" ]]; then
  COMMON_RUN_ARGS+=(--constant-temperature-test)
else
  COMMON_RUN_ARGS+=(--no-constant-temperature-test)
fi

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

YEAR=$((START_YEAR + PBS_ARRAY_INDEX))
if (( YEAR > END_YEAR )); then
  echo "[error] Computed YEAR=${YEAR} exceeds END_YEAR=${END_YEAR}" >&2
  exit 1
fi

TIME_START=$(printf "%04d-%sT00:00:00" "${YEAR}" "${RUN_START_MONTH_DAY}")
TIME_END=$(printf "%04d-%sT23:00:00" "${YEAR}" "${RUN_END_MONTH_DAY}")

ensure_manifest

echo "[info] $(date -Is) starting production year ${YEAR} on host $(hostname)"
echo "[info] output dir: ${PRODUCTION_OUTPUT_DIR}"
/usr/bin/time -v python run_budget.py \
  "${COMMON_RUN_ARGS[@]}" \
  --time-start "${TIME_START}" \
  --time-end "${TIME_END}"
echo "[info] $(date -Is) finished production year ${YEAR}"
