#!/bin/bash
#SBATCH --job-name=eulerian_heat_budget_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=24:00:00
#SBATCH --array=0-85%7
#SBATCH --output=/dev/null

set -euo pipefail

resolve_repo_root() {
  if [[ -n "${PROJECT_ROOT:-}" ]]; then
    cd "${PROJECT_ROOT}" && pwd
    return
  fi

  local submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
  if [[ -d "${submit_dir}/scripts" && -d "${submit_dir}/schedulers" ]]; then
    cd "${submit_dir}" && pwd
    return
  fi
  if [[ -d "${submit_dir}/../scripts" && -d "${submit_dir}/../schedulers" ]]; then
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

default_production_output_dir() {
  if [[ -n "${SCRATCH:-}" ]]; then
    printf "%s\n" "${SCRATCH}/eulerian_heat_budget/results/production/pnw_full_run_700_500_hPa"
  else
    printf "%s\n" "${REPO_ROOT}/results/production/pnw_full_run_700_500_hPa"
  fi
}

REPO_ROOT=$(resolve_repo_root)
SCRIPT_DIR="${REPO_ROOT}/scripts"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${JOB_ID}}"
ARRAY_TASK_ID_FOR_LOG="${SLURM_ARRAY_TASK_ID:-noarray}"
LOGFILE="${LOG_DIR}/${ARRAY_JOB_ID}_${ARRAY_TASK_ID_FOR_LOG}_EHB_prod.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOGFILE}") 2>&1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export EHB_DASK_N_WORKERS="${EHB_DASK_N_WORKERS:-4}"

if [[ -z "${HOME:-}" ]]; then
  HOME=$(getent passwd "$(id -un)" | cut -d: -f6)
  export HOME
fi
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/miniconda3}"
export EHB_CONDA_ENV="${EHB_CONDA_ENV:-dev_env}"

# source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
# mamba activate "${EHB_CONDA_ENV}"

# module load python/3.11
# virtualenv --no-download $SLURM_TMPDIR/env
source /home/mhpereir/conda_envs/ENV/bin/activate
# pip install --no-index --upgrade pip

# pip install --no-index -r /home/mhpereir/conda_envs/dev_env_nostats_requirements.txt

START_YEAR="${START_YEAR:-1940}"
END_YEAR="${END_YEAR:-2025}"
DATA_SOURCE="${DATA_SOURCE:-staged_arco_cache}"
DEFAULT_PRODUCTION_OUTPUT_DIR=$(default_production_output_dir)
PRODUCTION_OUTPUT_DIR="${PRODUCTION_OUTPUT_DIR:-${DEFAULT_PRODUCTION_OUTPUT_DIR}}"
REGION="${REGION:-pnw_bartusek}"
ZG_TOP_PA="${ZG_TOP_PA:-50000}"
ZG_BOTTOM_PA="${ZG_BOTTOM_PA:-70000}"
USE_SURFACE_AS_BOTTOM="${USE_SURFACE_AS_BOTTOM:-0}"
INIT_MANIFEST_ONLY="${INIT_MANIFEST_ONLY:-0}"
ENABLE_DIAGNOSTIC_PLOTS="${ENABLE_DIAGNOSTIC_PLOTS:-1}"
ENABLE_CONSTANT_TEMPERATURE_TEST="${ENABLE_CONSTANT_TEMPERATURE_TEST:-0}"
ENABLE_BENCHMARK_VARIABLES="${ENABLE_BENCHMARK_VARIABLES:-0}"
RUN_START_MONTH_DAY="${RUN_START_MONTH_DAY:-05-01}"
RUN_END_MONTH_DAY="${RUN_END_MONTH_DAY:-10-31}"
MANIFEST_PATH="${PRODUCTION_OUTPUT_DIR}/production_run.json"
MANIFEST_LOCK_DIR="${PRODUCTION_OUTPUT_DIR}/.manifest_init.lock"
MANIFEST_WAIT_SECONDS="${MANIFEST_WAIT_SECONDS:-300}"

if [[ "${DATA_SOURCE}" == "staged_arco_cache" && -z "${STAGED_CACHE_ROOT:-}" ]]; then
  echo "[error] DATA_SOURCE=staged_arco_cache requires STAGED_CACHE_ROOT to point to a local indexed cache root." >&2
  echo "[error] Populate it first with scripts/staged_arco_retrieval.py from an internet-capable session." >&2
  exit 1
fi

mkdir -p "${PRODUCTION_OUTPUT_DIR}"

cd "${SCRIPT_DIR}"

COMMON_RUN_ARGS=(
  --data-source "${DATA_SOURCE}"
  --production-output-dir "${PRODUCTION_OUTPUT_DIR}"
  --region "${REGION}"
  --zg-top-pa "${ZG_TOP_PA}"
)

if [[ "${DATA_SOURCE}" == "staged_arco_cache" ]]; then
  COMMON_RUN_ARGS+=(--staged-cache-root "${STAGED_CACHE_ROOT}")
fi

if [[ "${USE_SURFACE_AS_BOTTOM}" == "1" ]]; then
  COMMON_RUN_ARGS+=(--zg-bottom surface_pressure)
else
  COMMON_RUN_ARGS+=(
    --zg-bottom pressure_level
    --zg-bottom-pa "${ZG_BOTTOM_PA}"
  )
fi

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

if [[ "${ENABLE_BENCHMARK_VARIABLES}" == "1" ]]; then
  COMMON_RUN_ARGS+=(--include-benchmark-variables)
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
      else
        local status=$?
        rmdir "${MANIFEST_LOCK_DIR}" || true
        return "${status}"
      fi
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

echo "[info] repo root: ${REPO_ROOT}"
echo "[info] slurm job id: ${SLURM_JOB_ID:-not-set}"
echo "[info] slurm array job/task: ${SLURM_ARRAY_JOB_ID:-not-set}/${SLURM_ARRAY_TASK_ID:-not-set}"
echo "[info] dask: threaded scheduler, workers=${EHB_DASK_N_WORKERS}"
echo "[info] data source: ${DATA_SOURCE}"
if [[ "${DATA_SOURCE}" == "staged_arco_cache" ]]; then
  echo "[info] staged cache root: ${STAGED_CACHE_ROOT}"
fi

if [[ "${INIT_MANIFEST_ONLY}" == "1" ]]; then
  ensure_manifest
  exit 0
fi

: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set for yearly production runs}"

YEAR=$((START_YEAR + 10#${SLURM_ARRAY_TASK_ID}))
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
