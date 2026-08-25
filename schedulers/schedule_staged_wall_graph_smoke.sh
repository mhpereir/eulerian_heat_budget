#!/bin/bash
#PBS -N ehb_wall_graph_smoke
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=00:15:00
#PBS -j oe
#PBS -o /dev/null

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
LOG_DIR="${LOG_DIR:-${HOME:?HOME must be set}/eulerian-heat-budget/campaign-data/logs/development/staged-wall-graph-smoke}"
mkdir -p "${LOG_DIR}"
LOGFILE="${LOG_DIR}/${JOB_ID}_staged_wall_graph_smoke.log"
exec > >(tee -a "${LOGFILE}") 2>&1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
VENUS_MAMBA_ENV="${VENUS_MAMBA_ENV:-dev_env}"
mamba activate "${VENUS_MAMBA_ENV}"
PYTHON_EXECUTABLE=$(command -v python)

export MPLCONFIGDIR="${TMPDIR:-/tmp}/ehb-matplotlib-${JOB_ID%%.*}"
mkdir -p "${MPLCONFIGDIR}"

cd "${PROJECT_ROOT}"
echo "[info] $(date -Is) starting staged-wall graph smoke on $(hostname)"
echo "[info] repo root: ${PROJECT_ROOT}"
echo "[info] expected commit: ${EXPECTED_COMMIT}"
echo "[info] Venus Mamba environment: ${VENUS_MAMBA_ENV}"
echo "[info] Python executable: ${PYTHON_EXECUTABLE}"
echo "[info] log file: ${LOGFILE}"

/usr/bin/time -v "${PYTHON_EXECUTABLE}" -m pytest -q -W error \
    tests/test_arco_cache.py \
    tests/test_plot_diagnostics.py \
    tests/test_validate_budget_artifacts.py \
    -k "expand_sparse_wall or reconstruct_budget_dataset_keeps_canonical_shape or reconstruct_benchmark_dataset_expands_compact_shell or empty_quantile_bins or nanmean_or_nan or validate_run"

echo "[info] $(date -Is) staged-wall graph smoke passed"
