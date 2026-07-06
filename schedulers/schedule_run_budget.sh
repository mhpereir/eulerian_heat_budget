#!/bin/bash
#PBS -N eulerian_head_budget
#PBS -l select=1:ncpus=12:mem=32gb
#PBS -j oe
#PBS -o /home/mhpereir/eulerian_heat_budget/logs/

set -euo pipefail

JOB_ID="${PBS_JOBID:-manual}"
LOG_DIR="${LOG_DIR:-/home/mhpereir/eulerian_heat_budget/logs}"
mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/${JOB_ID}_EHB_single.log"
exec > >(tee -a "${LOGFILE}") 2>&1

PROJECT_ROOT="${PROJECT_ROOT:-/home/mhpereir/eulerian_heat_budget}"
SCHEDULER_DIR="${SCHEDULER_DIR:-${PROJECT_ROOT}/schedulers}"
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/scripts}"
SETTINGS_FILE="${SINGLE_RUN_CLI_SETTINGS:-${SCHEDULER_DIR}/single_run_cli_settings}"

source "${SETTINGS_FILE}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${EHB_CONDA_ENV:-dev_env}"

RUN_ARGS=()
ehb_build_run_budget_args RUN_ARGS

cd "${SCRIPT_DIR}"

echo "[info] $(date -Is) starting eulerian heat budget calculation on host $(hostname)"
echo "[info] repo root: ${PROJECT_ROOT}"
echo "[info] log file: ${LOGFILE}"
echo "[info] settings file: ${SETTINGS_FILE}"
echo "[info] data source: ${DATA_SOURCE}"
if [[ "${DATA_SOURCE}" == "staged_arco_cache" ]]; then
  echo "[info] staged cache root: ${STAGED_CACHE_ROOT}"
fi

/usr/bin/time -v python run_budget.py "${RUN_ARGS[@]}"
echo "[info] $(date -Is) done"
