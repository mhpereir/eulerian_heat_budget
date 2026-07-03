#!/bin/bash
#PBS -N ehb_stage_arco
#PBS -l select=1:ncpus=8:mem=24gb
#PBS -j oe
#PBS -o /dev/null
# PBS -o /home/mhpereir/eulerian_heat_budget/logs/

set -euo pipefail

SCHEDULER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCHEDULER_DIR}/.." && pwd)}"
SCRIPT_DIR="${PROJECT_ROOT}/scripts"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
JOB_ID="${PBS_JOBID:-manual}"

source "${SCHEDULER_DIR}/single_run_cli_settings"

mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/${JOB_ID}_EHB_stage_arco.log"
exec > >(tee -a "${LOGFILE}") 2>&1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${EHB_CONDA_ENV:-dev_env}"

RETRIEVAL_ARGS=()
ehb_build_staged_arco_retrieval_args RETRIEVAL_ARGS

cd "${SCRIPT_DIR}"

echo "[info] $(date -Is) starting staged ARCO retrieval on host $(hostname)"
echo "[info] repo root: ${PROJECT_ROOT}"
echo "[info] staged cache root: ${STAGED_CACHE_ROOT}"

/usr/bin/time -v python staged_arco_retrieval.py "${RETRIEVAL_ARGS[@]}"
echo "[info] $(date -Is) done"
