#!/bin/bash
#PBS -N ehb_stage_arco_prod
#PBS -J 0-85%5
#PBS -l select=1:ncpus=8:mem=8gb
#PBS -l walltime=48:00:00
#PBS -j oe

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be supplied by the submission workflow}"
JOB_ID="${PBS_JOBID:-manual}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/${JOB_ID}_EHB_stage_arco_prod.log"
exec > >(tee -a "${LOGFILE}") 2>&1

SCHEDULER_DIR="${SCHEDULER_DIR:-${PROJECT_ROOT}/schedulers}"
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/scripts}"
SETTINGS_FILE="${PRODUCTION_RUN_CLI_SETTINGS:-${SCHEDULER_DIR}/production_run_cli_settings.sh}"

source "${SETTINGS_FILE}"
ehb_verify_runtime_checkout
ehb_require_external_production_paths
ehb_require_staged_run_manifest

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

: "${PBS_ARRAY_INDEX:?PBS_ARRAY_INDEX must be set for yearly production staged ARCO retrieval}"

YEAR=$(ehb_production_year_for_task "${PBS_ARRAY_INDEX}")
ehb_validate_production_year "${YEAR}"
ehb_build_production_time_window "${YEAR}" TIME_START TIME_END
YEAR_CACHE_ROOT=$(ehb_year_shard_root "${YEAR}")

CAMPAIGN_ARGS=()
ehb_build_staged_campaign_init_args CAMPAIGN_ARGS

RETRIEVAL_ARGS=()
ehb_build_production_staged_retrieval_args \
  RETRIEVAL_ARGS "${TIME_START}" "${TIME_END}" "${YEAR_CACHE_ROOT}"

cd "${SCRIPT_DIR}"

echo "[info] $(date -Is) starting production staged ARCO retrieval for year ${YEAR} on host $(hostname)"
echo "[info] repo root: ${PROJECT_ROOT}"
echo "[info] log file: ${LOGFILE}"
echo "[info] settings file: ${SETTINGS_FILE}"
echo "[info] expected commit: ${EXPECTED_COMMIT}"
echo "[info] Venus Mamba environment: ${VENUS_MAMBA_ENV}"
echo "[info] Python executable: ${PYTHON_EXECUTABLE}"
echo "[info] campaign cache root: ${STAGED_CACHE_ROOT}"
echo "[info] yearly shard root: ${YEAR_CACHE_ROOT}"
echo "[info] time window: ${TIME_START} to ${TIME_END}"

"${PYTHON_EXECUTABLE}" -u staged_arco_campaign.py "${CAMPAIGN_ARGS[@]}"
/usr/bin/time -v "${PYTHON_EXECUTABLE}" -u staged_arco_retrieval.py "${RETRIEVAL_ARGS[@]}"
"${PYTHON_EXECUTABLE}" -u staged_arco_campaign.py finalize-year \
  --cache-root "${STAGED_CACHE_ROOT}" \
  --year "${YEAR}" \
  --pbs-job-id "${PBS_JOBID:-}" \
  --pbs-array-index "${PBS_ARRAY_INDEX}" \
  --git-commit "${EXPECTED_COMMIT}"
echo "[info] $(date -Is) finished production staged ARCO retrieval for year ${YEAR}"
