#!/bin/bash
#PBS -N pbl_check
#PBS -J 0-85
#PBS -l select=1:ncpus=4:mem=24gb
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail

SCHEDULER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCHEDULER_DIR}/.." && pwd)
SCRIPT_DIR="${REPO_ROOT}/scripts"
LOG_DIR="${REPO_ROOT}/logs"

mkdir -p "${LOG_DIR}"

LOGFILE="${LOG_DIR}/pbl_check_${PBS_JOBID:-manual}.log"
exec > >(tee -a "${LOGFILE}") 2>&1

export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate dev_env

START_YEAR=1940
END_YEAR=2025
BBOX=("${BBOX_LAT_MIN:-60}" "${BBOX_LAT_MAX:-40}" "${BBOX_LON_MIN:--130}" "${BBOX_LON_MAX:--110}")

: "${PBS_ARRAY_INDEX:?PBS_ARRAY_INDEX must be set for yearly check_pbl runs}"

YEAR=$((START_YEAR + PBS_ARRAY_INDEX))
if (( YEAR > END_YEAR )); then
  echo "[error] Computed YEAR=${YEAR} exceeds END_YEAR=${END_YEAR}" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"

echo "[info] $(date -Is) starting year ${YEAR} on host $(hostname)"
/usr/bin/time -v python check_pbl.py \
        --year "${YEAR}" \
        --bbox "${BBOX[@]}"
echo "[info] $(date -Is) finished year ${YEAR}"
