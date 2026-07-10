#!/bin/bash
#PBS -N ehb_stage_arco_prod
#PBS -J 0-85%5
#PBS -l select=1:ncpus=8:mem=24gb
#PBS -j oe
#PBS -o /home/mhpereir/eulerian_heat_budget/logs/

set -euo pipefail

JOB_ID="${PBS_JOBID:-manual}"
LOG_DIR="${LOG_DIR:-/home/mhpereir/eulerian_heat_budget/logs}"
mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/${JOB_ID}_EHB_stage_arco_prod.log"
exec > >(tee -a "${LOGFILE}") 2>&1

PROJECT_ROOT="${PROJECT_ROOT:-/home/mhpereir/eulerian_heat_budget}"
SCHEDULER_DIR="${SCHEDULER_DIR:-${PROJECT_ROOT}/schedulers}"
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/scripts}"
SETTINGS_FILE="${PRODUCTION_RUN_CLI_SETTINGS:-${SCHEDULER_DIR}/production_run_cli_settings.sh}"

source "${SETTINGS_FILE}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${EHB_CONDA_ENV:-dev_env}"

: "${PBS_ARRAY_INDEX:?PBS_ARRAY_INDEX must be set for yearly production staged ARCO retrieval}"

YEAR=$(ehb_production_year_for_task "${PBS_ARRAY_INDEX}")
ehb_validate_production_year "${YEAR}"
ehb_build_production_time_window "${YEAR}" TIME_START TIME_END

RETRIEVAL_ARGS=()
ehb_build_production_staged_retrieval_args RETRIEVAL_ARGS "${TIME_START}" "${TIME_END}"

cd "${SCRIPT_DIR}"

echo "[info] $(date -Is) starting production staged ARCO retrieval for year ${YEAR} on host $(hostname)"
echo "[info] repo root: ${PROJECT_ROOT}"
echo "[info] log file: ${LOGFILE}"
echo "[info] settings file: ${SETTINGS_FILE}"
echo "[info] staged cache root: ${STAGED_CACHE_ROOT}"
echo "[info] time window: ${TIME_START} to ${TIME_END}"

/usr/bin/time -v python -u staged_arco_retrieval.py "${RETRIEVAL_ARGS[@]}"
echo "[info] $(date -Is) finished production staged ARCO retrieval for year ${YEAR}"
