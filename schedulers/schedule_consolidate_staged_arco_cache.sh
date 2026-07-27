#!/bin/bash
#PBS -N ehb_consolidate_arco
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=12:00:00
#PBS -j oe

set -euo pipefail

JOB_ID="${PBS_JOBID:-manual}"
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be supplied by the submission workflow}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/${JOB_ID}_EHB_consolidate_arco.log"
exec > >(tee -a "${LOGFILE}") 2>&1

SCHEDULER_DIR="${SCHEDULER_DIR:-${PROJECT_ROOT}/schedulers}"
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/scripts}"
SETTINGS_FILE="${PRODUCTION_RUN_CLI_SETTINGS:-${SCHEDULER_DIR}/production_run_cli_settings.sh}"

source "${SETTINGS_FILE}"
ehb_verify_runtime_checkout
ehb_require_external_production_paths
ehb_require_staged_cache_root "staged ARCO shard consolidation"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
VENUS_MAMBA_ENV="${VENUS_MAMBA_ENV:-dev_env}"
mamba activate "${VENUS_MAMBA_ENV}"
PYTHON_EXECUTABLE=$(command -v python)

cd "${SCRIPT_DIR}"

echo "[info] $(date -Is) starting staged ARCO consolidation on host $(hostname)"
echo "[info] repo root: ${PROJECT_ROOT}"
echo "[info] expected commit: ${EXPECTED_COMMIT}"
echo "[info] log file: ${LOGFILE}"
echo "[info] settings file: ${SETTINGS_FILE}"
echo "[info] campaign cache root: ${STAGED_CACHE_ROOT}"
echo "[info] Venus Mamba environment: ${VENUS_MAMBA_ENV}"
echo "[info] Python executable: ${PYTHON_EXECUTABLE}"

/usr/bin/time -v "${PYTHON_EXECUTABLE}" -u consolidate_staged_arco_cache.py \
  --cache-root "${STAGED_CACHE_ROOT}"
echo "[info] $(date -Is) finished staged ARCO consolidation"
