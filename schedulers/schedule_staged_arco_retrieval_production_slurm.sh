#!/bin/bash
#SBATCH --job-name=ehb_stage_arco_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-74
# Submit from the repository root after ensuring logs/ exists; Slurm
# resolves output paths before this script can compute REPO_ROOT.
#SBATCH --output=logs/%A_%a_EHB_stage_arco_prod.log
#SBATCH --error=logs/%A_%a_EHB_stage_arco_prod.log

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

REPO_ROOT=$(resolve_repo_root)
SCRIPT_DIR="${REPO_ROOT}/scripts"
SCHEDULER_DIR="${REPO_ROOT}/schedulers"
SETTINGS_FILE="${PRODUCTION_RUN_CLI_SETTINGS:-${SCHEDULER_DIR}/production_run_cli_settings.sh}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${JOB_ID}}"
ARRAY_TASK_ID_FOR_LOG="${SLURM_ARRAY_TASK_ID:-noarray}"
LOGFILE="${LOG_DIR}/${ARRAY_JOB_ID}_${ARRAY_TASK_ID_FOR_LOG}_EHB_stage_arco_prod.log"

mkdir -p "${LOG_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  exec > >(tee -a "${LOGFILE}" >/dev/null) 2>&1
else
  exec > >(tee -a "${LOGFILE}") 2>&1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

EHB_DASK_MAX_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
export EHB_DASK_N_WORKERS="${EHB_DASK_N_WORKERS:-${EHB_DASK_MAX_WORKERS}}"

source "${SETTINGS_FILE}"

ehb_require_staged_cache_root "production staged ARCO retrieval"

: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set for yearly production staged ARCO retrieval}"

YEAR=$(ehb_production_year_for_task "${SLURM_ARRAY_TASK_ID}")
ehb_validate_production_year "${YEAR}"
ehb_build_production_time_window "${YEAR}" TIME_START TIME_END

RETRIEVAL_ARGS=()
ehb_build_production_staged_retrieval_args RETRIEVAL_ARGS "${TIME_START}" "${TIME_END}"

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

cd "${SCRIPT_DIR}"

echo "[info] $(date -Is) starting production staged ARCO retrieval for year ${YEAR} on host $(hostname)"
echo "[info] repo root: ${REPO_ROOT}"
echo "[info] slurm job id: ${SLURM_JOB_ID:-not-set}"
echo "[info] slurm array job/task: ${SLURM_ARRAY_JOB_ID:-not-set}/${SLURM_ARRAY_TASK_ID:-not-set}"
echo "[info] dask: threaded scheduler, requested_workers=${EHB_DASK_N_WORKERS}, max_workers=${EHB_DASK_MAX_WORKERS}"
echo "[info] settings file: ${SETTINGS_FILE}"
echo "[info] staged cache root: ${STAGED_CACHE_ROOT}"
echo "[info] time window: ${TIME_START} to ${TIME_END}"

python -u staged_arco_retrieval.py "${RETRIEVAL_ARGS[@]}"
echo "[info] $(date -Is) finished production staged ARCO retrieval for year ${YEAR}"
