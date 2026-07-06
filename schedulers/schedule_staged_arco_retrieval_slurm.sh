#!/bin/bash
#SBATCH --job-name=ehb_stage_arco
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=24:00:00
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

REPO_ROOT=$(resolve_repo_root)
SCRIPT_DIR="${REPO_ROOT}/scripts"
SCHEDULER_DIR="${REPO_ROOT}/schedulers"
SETTINGS_FILE="${SINGLE_RUN_CLI_SETTINGS:-${SCHEDULER_DIR}/single_run_cli_settings}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LOGFILE="${LOG_DIR}/${JOB_ID}_EHB_stage_arco.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOGFILE}") 2>&1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export EHB_DASK_N_WORKERS="${EHB_DASK_N_WORKERS:-4}"

source "${SETTINGS_FILE}"

RETRIEVAL_ARGS=()
ehb_build_staged_arco_retrieval_args RETRIEVAL_ARGS

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

echo "[info] $(date -Is) starting staged ARCO retrieval on host $(hostname)"
echo "[info] repo root: ${REPO_ROOT}"
echo "[info] slurm job id: ${SLURM_JOB_ID:-not-set}"
echo "[info] dask: threaded scheduler, workers=${EHB_DASK_N_WORKERS}"
echo "[info] settings file: ${SETTINGS_FILE}"
echo "[info] staged cache root: ${STAGED_CACHE_ROOT}"

python staged_arco_retrieval.py "${RETRIEVAL_ARGS[@]}"
echo "[info] $(date -Is) done"
