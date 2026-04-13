#!/bin/bash
#PBS -N pbl_check
#PBS -l select=1:ncpus=4:mem=12gb
#PBS -j oe
#PBS -o /dev/null
# PBS -o /home/mhpereir/eulerian_heat_budget/logs/

LOGFILE="/home/mhpereir/eulerian_heat_budget/logs/pbl_check_${PBS_JOBID}.log"
exec > >(tee -a "${LOGFILE}") 2>&1

export MAMBA_ROOT_PREFIX=/home/mhpereir/miniconda3
source /home/mhpereir/miniconda3/etc/profile.d/mamba.sh
mamba activate dev_env

set -euo pipefail

cd /home/mhpereir/eulerian_heat_budget/scripts

echo "[info] $(date -Is) starting on host $(hostname)"
/usr/bin/time -v python check_pbl.py \
        --year-start 2021 \
        --year-end 2021 \
        --bbox 60 40 -130 -110
echo "[info] $(date -Is) done"
