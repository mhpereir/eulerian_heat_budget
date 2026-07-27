#!/bin/bash
# Submit yearly staged retrieval and its dependent consolidation job.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must identify the clean Venus checkout}"
SCHEDULER_DIR="${SCHEDULER_DIR:-${PROJECT_ROOT}/schedulers}"
SETTINGS_FILE="${PRODUCTION_RUN_CLI_SETTINGS:-${SCHEDULER_DIR}/production_run_cli_settings.sh}"
QSUB_BIN="${QSUB_BIN:-/opt/pbs/bin/qsub}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"

source "${SETTINGS_FILE}"
ehb_require_external_production_paths

if [[ ! -x "${QSUB_BIN}" ]]; then
  echo "[error] qsub is not executable: ${QSUB_BIN}" >&2
  exit 2
fi
if (( MAX_PARALLEL < 1 )); then
  echo "[error] MAX_PARALLEL must be at least one." >&2
  exit 2
fi
mkdir -p "${LOG_DIR}"
if [[ ! -d "${LOG_DIR}" || ! -w "${LOG_DIR}" ]]; then
  echo "[error] PBS log directory is not writable: ${LOG_DIR}" >&2
  exit 2
fi

EXPECTED_COMMIT=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
UPSTREAM_COMMIT=$(git -C "${PROJECT_ROOT}" rev-parse '@{u}')
if [[ "${EXPECTED_COMMIT}" != "${UPSTREAM_COMMIT}" ]]; then
  echo "[error] Refusing to submit an unpushed commit: ${EXPECTED_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)" ]]; then
  echo "[error] Refusing to submit from a dirty checkout: ${PROJECT_ROOT}" >&2
  exit 2
fi

TASK_COUNT=$((END_YEAR - START_YEAR + 1))
if (( TASK_COUNT < 1 )); then
  echo "[error] END_YEAR must not precede START_YEAR." >&2
  exit 2
fi
LAST_TASK=$((TASK_COUNT - 1))
EXPORTS="PROJECT_ROOT=${PROJECT_ROOT},EXPECTED_COMMIT=${EXPECTED_COMMIT},PRODUCTION_RUN_CLI_SETTINGS=${SETTINGS_FILE}"
EXPORTS+=",CAMPAIGN_ID=${CAMPAIGN_ID},START_YEAR=${START_YEAR},END_YEAR=${END_YEAR}"
EXPORTS+=",RUN_START_MONTH_DAY=${RUN_START_MONTH_DAY},RUN_END_MONTH_DAY=${RUN_END_MONTH_DAY}"
EXPORTS+=",REGION=${REGION},MARGIN_N=${MARGIN_N},ZG_TOP_PA=${ZG_TOP_PA},ZG_BOTTOM=${ZG_BOTTOM}"
EXPORTS+=",ZG_BOTTOM_PA=${ZG_BOTTOM_PA},ALLOW_BOTTOM_OVERFLOW=${ALLOW_BOTTOM_OVERFLOW}"
EXPORTS+=",ENABLE_BENCHMARK_VARIABLES=${ENABLE_BENCHMARK_VARIABLES}"
EXPORTS+=",STAGED_CACHE_BASE_ROOT=${STAGED_CACHE_BASE_ROOT},STAGED_CACHE_ROOT=${STAGED_CACHE_ROOT}"
EXPORTS+=",STAGED_ARCO_TIME_CHUNK=${STAGED_ARCO_TIME_CHUNK}"
EXPORTS+=",STAGED_ARCO_ATTEMPT_TIMEOUT_SECONDS=${STAGED_ARCO_ATTEMPT_TIMEOUT_SECONDS}"
EXPORTS+=",VENUS_MAMBA_ENV=${VENUS_MAMBA_ENV:-dev_env}"
EXPORTS+=",LOG_DIR=${LOG_DIR}"
EXPORTS+=",EHB_WORKSPACE_ROOT=${EHB_WORKSPACE_ROOT},EHB_CAMPAIGN_DATA_ROOT=${EHB_CAMPAIGN_DATA_ROOT}"
EXPORTS+=",EHB_STAGED_ZARR_ROOT=${EHB_STAGED_ZARR_ROOT},EHB_RUN_BUDGET_ROOT=${EHB_RUN_BUDGET_ROOT}"
EXPORTS+=",EHB_LOG_ROOT=${EHB_LOG_ROOT},RUN_ID=${RUN_ID},PRODUCTION_OUTPUT_DIR=${PRODUCTION_OUTPUT_DIR}"

echo "[submit] campaign=${CAMPAIGN_ID}"
echo "[submit] cache_root=${STAGED_CACHE_ROOT}"
echo "[submit] commit=${EXPECTED_COMMIT}"
echo "[submit] log_dir=${LOG_DIR}"
echo "[submit] retrieval_array=0-${LAST_TASK}%${MAX_PARALLEL}"

RETRIEVAL_JOB_ID=$(
  "${QSUB_BIN}" \
    -J "0-${LAST_TASK}%${MAX_PARALLEL}" \
    -o "${LOG_DIR}/" \
    -v "${EXPORTS}" \
    "${SCHEDULER_DIR}/schedule_staged_arco_retrieval_production.sh"
)
echo "[submit] retrieval_job_id=${RETRIEVAL_JOB_ID}"

CONSOLIDATION_JOB_ID=$(
  "${QSUB_BIN}" \
    -W "depend=afterok:${RETRIEVAL_JOB_ID}" \
    -o "${LOG_DIR}/" \
    -v "${EXPORTS}" \
    "${SCHEDULER_DIR}/schedule_consolidate_staged_arco_cache.sh"
)
echo "[submit] consolidation_job_id=${CONSOLIDATION_JOB_ID}"
