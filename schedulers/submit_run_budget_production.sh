#!/bin/bash
# Submit the yearly production heat-budget array.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must identify the clean Venus checkout}"
SCHEDULER_DIR="${SCHEDULER_DIR:-${PROJECT_ROOT}/schedulers}"
SETTINGS_FILE="${PRODUCTION_RUN_CLI_SETTINGS:-${SCHEDULER_DIR}/production_run_cli_settings.sh}"
QSUB_BIN="${QSUB_BIN:-/opt/pbs/bin/qsub}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"

source "${SETTINGS_FILE}"
ehb_require_external_production_paths
EXPECTED_COMMIT=$(ehb_verify_production_submission_checkout)
ehb_require_consolidated_staged_cache

if [[ ! -x "${QSUB_BIN}" ]]; then
  echo "[error] qsub is not executable: ${QSUB_BIN}" >&2
  exit 2
fi
if (( MAX_PARALLEL < 1 )); then
  echo "[error] MAX_PARALLEL must be at least one." >&2
  exit 2
fi
mkdir -p "${LOG_DIR}" "${PRODUCTION_OUTPUT_DIR}"
if [[ ! -w "${LOG_DIR}" || ! -w "${PRODUCTION_OUTPUT_DIR}" ]]; then
  echo "[error] production log or output directory is not writable." >&2
  exit 2
fi

TASK_COUNT=$((END_YEAR - START_YEAR + 1))
if (( TASK_COUNT < 1 )); then
  echo "[error] END_YEAR must not precede START_YEAR." >&2
  exit 2
fi
LAST_TASK=$((TASK_COUNT - 1))

EXPORTS="PROJECT_ROOT=${PROJECT_ROOT},EXPECTED_COMMIT=${EXPECTED_COMMIT},PRODUCTION_RUN_CLI_SETTINGS=${SETTINGS_FILE}"
EXPORTS+=",START_YEAR=${START_YEAR},END_YEAR=${END_YEAR},RUN_START_MONTH_DAY=${RUN_START_MONTH_DAY}"
EXPORTS+=",RUN_END_MONTH_DAY=${RUN_END_MONTH_DAY},DATA_SOURCE=${DATA_SOURCE}"
EXPORTS+=",CAMPAIGN_ID=${CAMPAIGN_ID},RUN_ID=${RUN_ID},REGION=${REGION},MARGIN_N=${MARGIN_N}"
EXPORTS+=",ZG_TOP_PA=${ZG_TOP_PA},ZG_BOTTOM=${ZG_BOTTOM},ZG_BOTTOM_PA=${ZG_BOTTOM_PA}"
EXPORTS+=",ALLOW_BOTTOM_OVERFLOW=${ALLOW_BOTTOM_OVERFLOW}"
EXPORTS+=",ENABLE_DIAGNOSTIC_PLOTS=${ENABLE_DIAGNOSTIC_PLOTS}"
EXPORTS+=",ENABLE_CONSTANT_TEMPERATURE_TEST=${ENABLE_CONSTANT_TEMPERATURE_TEST}"
EXPORTS+=",ENABLE_BENCHMARK_VARIABLES=${ENABLE_BENCHMARK_VARIABLES}"
EXPORTS+=",STAGED_CACHE_ROOT=${STAGED_CACHE_ROOT},PRODUCTION_OUTPUT_DIR=${PRODUCTION_OUTPUT_DIR}"
EXPORTS+=",MANIFEST_PATH=${MANIFEST_PATH},MANIFEST_LOCK_DIR=${MANIFEST_LOCK_DIR}"
EXPORTS+=",MANIFEST_WAIT_SECONDS=${MANIFEST_WAIT_SECONDS},VENUS_MAMBA_ENV=${VENUS_MAMBA_ENV:-dev_env}"
EXPORTS+=",LOG_DIR=${LOG_DIR},EHB_WORKSPACE_ROOT=${EHB_WORKSPACE_ROOT}"
EXPORTS+=",EHB_CAMPAIGN_DATA_ROOT=${EHB_CAMPAIGN_DATA_ROOT}"
EXPORTS+=",EHB_STAGED_ZARR_ROOT=${EHB_STAGED_ZARR_ROOT}"
EXPORTS+=",EHB_RUN_BUDGET_ROOT=${EHB_RUN_BUDGET_ROOT},EHB_LOG_ROOT=${EHB_LOG_ROOT}"

echo "[submit] run_id=${RUN_ID}"
echo "[submit] cache_root=${STAGED_CACHE_ROOT}"
echo "[submit] output_dir=${PRODUCTION_OUTPUT_DIR}"
echo "[submit] log_dir=${LOG_DIR}"
echo "[submit] commit=${EXPECTED_COMMIT}"
echo "[submit] production_array=0-${LAST_TASK}%${MAX_PARALLEL}"

PRODUCTION_JOB_ID=$(
  "${QSUB_BIN}" \
    -J "0-${LAST_TASK}%${MAX_PARALLEL}" \
    -o "${LOG_DIR}/" \
    -v "${EXPORTS}" \
    "${SCHEDULER_DIR}/schedule_run_budget_production.sh"
)
echo "[submit] production_job_id=${PRODUCTION_JOB_ID}"
