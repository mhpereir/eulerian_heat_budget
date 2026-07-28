#!/bin/bash
#PBS -N ehb_stage_arco
#PBS -l select=1:ncpus=8:mem=8gb
#PBS -l walltime=48:00:00
#PBS -j oe

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be supplied by the submission workflow}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT must be supplied by the submission workflow}"
PBS_WORKDIR="${PBS_O_WORKDIR:?PBS_O_WORKDIR is not set}"

PROJECT_ROOT=$(realpath -e -- "${PROJECT_ROOT}")
PBS_WORKDIR=$(realpath -e -- "${PBS_WORKDIR}")
if [[ "${PBS_WORKDIR}" != "${PROJECT_ROOT}" ]]; then
    echo "[error] PBS_O_WORKDIR must be the project Git root: ${PROJECT_ROOT}" >&2
    exit 2
fi

ACTUAL_COMMIT=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
    echo "[error] checkout commit ${ACTUAL_COMMIT} does not match ${EXPECTED_COMMIT}" >&2
    exit 2
fi
if [[ -n "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)" ]]; then
    echo "[error] runtime checkout is dirty: ${PROJECT_ROOT}" >&2
    exit 2
fi

JOB_ID="${PBS_JOBID:-manual}"
LOG_DIR="${LOG_DIR:-${HOME:?HOME must be set}/eulerian-heat-budget/campaign-data/logs/development/single-run}"
mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/${JOB_ID}_EHB_stage_arco.log"
exec > >(tee -a "${LOGFILE}") 2>&1

SCHEDULER_DIR="${SCHEDULER_DIR:-${PROJECT_ROOT}/schedulers}"
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/scripts}"
SETTINGS_FILE="${SINGLE_RUN_CLI_SETTINGS:-${SCHEDULER_DIR}/single_run_cli_settings}"

source "${SETTINGS_FILE}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
VENUS_MAMBA_ENV="${VENUS_MAMBA_ENV:-${EHB_CONDA_ENV:-dev_env}}"
mamba activate "${VENUS_MAMBA_ENV}"
PYTHON_EXECUTABLE=$(command -v python)

RETRIEVAL_ARGS=()
ehb_build_staged_arco_retrieval_args RETRIEVAL_ARGS

cd "${PBS_WORKDIR}"

echo "[info] $(date -Is) starting staged ARCO retrieval on host $(hostname)"
echo "[info] repo root: ${PROJECT_ROOT}"
echo "[info] expected commit: ${EXPECTED_COMMIT}"
echo "[info] Venus Mamba environment: ${VENUS_MAMBA_ENV}"
echo "[info] Python executable: ${PYTHON_EXECUTABLE}"
echo "[info] log file: ${LOGFILE}"
echo "[info] settings file: ${SETTINGS_FILE}"
echo "[info] staged cache root: ${STAGED_CACHE_ROOT}"

/usr/bin/time -v "${PYTHON_EXECUTABLE}" -u \
    "${SCRIPT_DIR}/staged_arco_retrieval.py" "${RETRIEVAL_ARGS[@]}"
echo "[info] $(date -Is) done"
