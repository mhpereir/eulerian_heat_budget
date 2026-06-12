#!/bin/bash
#PBS -N eulerian_head_budget
#PBS -l select=1:ncpus=12:mem=32gb
#PBS -j oe
#PBS -o /dev/null
# PBS -o /home/mhpereir/eulerian_heat_budget/logs/

LOGFILE="/home/mhpereir/eulerian_heat_budget/logs/${PBS_JOBID}_EHB_single.log"
exec > >(tee -a "${LOGFILE}") 2>&1

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
REGION="${REGION:-ocean_test}"

cd /home/mhpereir/eulerian_heat_budget/scripts

echo "[info] $(date -Is) starting eulerian heat budget calculation on host $(hostname)"
/usr/bin/time -v python run_budget.py \
  --data-source arco_era5 \
  --region "${REGION}" \
  --time-start "${TIME_START}" \
  --time-end "${TIME_END}" \
  --zg-bottom "pressure_level" \
  --zg-bottom-pa 50000 \
  --zg-top-pa 30000 \
  --diagnostic-plots \
  --constant-temperature-test
echo "[info] $(date -Is) done"


  # --include-benchmark-variables \
