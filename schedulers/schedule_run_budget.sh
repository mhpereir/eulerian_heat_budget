#!/bin/bash
#PBS -N eulerian_head_budget
#PBS -l select=1:ncpus=12:mem=32gb
#PBS -j oe
#PBS -o /home/mhpereir/eulerian_heat_budget/logs/

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail

TIME_START="1941-06-01T00:00:00"
TIME_END="1941-06-07T00:00:00"

cd /home/mhpereir/eulerian_heat_budget/scripts

echo "[info] $(date -Is) starting eulerian heat budget calculation on host $(hostname)"
/usr/bin/time -v python run_budget.py \
  --data-source arco_era5 \
  --time-start "${TIME_START}" \
  --time-end "${TIME_END}" \
  --diagnostic-plots \
  --constant-temperature-test
echo "[info] $(date -Is) done"
