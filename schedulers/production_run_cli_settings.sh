# Shared CLI settings for PBS production ARCO staging and production budget runs.
#
# This file is sourced by:
# - schedulers/schedule_run_budget_production.sh
# - schedulers/schedule_staged_arco_retrieval_production.sh
#
# Keep production staging and compute settings here so they cannot drift.

START_YEAR="${START_YEAR:-1940}"
END_YEAR="${END_YEAR:-2025}"
RUN_START_MONTH_DAY="${RUN_START_MONTH_DAY:-05-01}"
RUN_END_MONTH_DAY="${RUN_END_MONTH_DAY:-10-31}"

DATA_SOURCE="${DATA_SOURCE:-staged_arco_cache}"
EHB_OUTPUT_ROOT="${EHB_OUTPUT_ROOT:-${PROJECT_ROOT:?PROJECT_ROOT must be set}/results}"
PRODUCTION_OUTPUT_DIR="${PRODUCTION_OUTPUT_DIR:-${EHB_OUTPUT_ROOT}/production/alaska_surface_700hPa_1940_2025}"
CAMPAIGN_ID="${CAMPAIGN_ID:-alaska-surface-700hpa-1940-2025}"
REGION="${REGION:-alaska}"
MARGIN_N="${MARGIN_N:-1}"
ZG_TOP_PA="${ZG_TOP_PA:-70000}"
ZG_BOTTOM="${ZG_BOTTOM:-surface_pressure}"
ZG_BOTTOM_PA="${ZG_BOTTOM_PA:-}"
ALLOW_BOTTOM_OVERFLOW="${ALLOW_BOTTOM_OVERFLOW:-1}"

STAGED_CACHE_BASE_ROOT="${STAGED_CACHE_BASE_ROOT:-${EHB_OUTPUT_ROOT}/staged_arco_cache}"
STAGED_CACHE_ROOT="${STAGED_CACHE_ROOT:-${STAGED_CACHE_BASE_ROOT}/${CAMPAIGN_ID}}"
STAGED_ARCO_TIME_CHUNK="${STAGED_ARCO_TIME_CHUNK:-month}"
STAGED_ARCO_ATTEMPT_TIMEOUT_SECONDS="${STAGED_ARCO_ATTEMPT_TIMEOUT_SECONDS:-20800}"

INIT_MANIFEST_ONLY="${INIT_MANIFEST_ONLY:-0}"
ENABLE_DIAGNOSTIC_PLOTS="${ENABLE_DIAGNOSTIC_PLOTS:-1}"
ENABLE_CONSTANT_TEMPERATURE_TEST="${ENABLE_CONSTANT_TEMPERATURE_TEST:-0}"
ENABLE_BENCHMARK_VARIABLES="${ENABLE_BENCHMARK_VARIABLES:-0}" # use only with full atmosphere
MANIFEST_PATH="${MANIFEST_PATH:-${PRODUCTION_OUTPUT_DIR}/production_run.json}"
MANIFEST_LOCK_DIR="${MANIFEST_LOCK_DIR:-${PRODUCTION_OUTPUT_DIR}/.manifest_init.lock}"
MANIFEST_WAIT_SECONDS="${MANIFEST_WAIT_SECONDS:-300}"

ehb_bool_enabled() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

ehb_require_staged_cache_root() {
  local context="$1"

  if [[ -z "${STAGED_CACHE_ROOT:-}" ]]; then
    echo "[error] ${context} requires STAGED_CACHE_ROOT to point to a local indexed cache root." >&2
    echo "[error] Populate it first with scripts/staged_arco_retrieval.py from an internet-capable session." >&2
    return 1
  fi
}

ehb_require_consolidated_staged_cache() {
  ehb_require_staged_cache_root "production heat-budget calculation"
  local required
  for required in campaign.json cache.sqlite consolidation.json; do
    if [[ ! -f "${STAGED_CACHE_ROOT}/${required}" ]]; then
      echo "[error] staged campaign is not consolidated: ${STAGED_CACHE_ROOT}/${required}" >&2
      return 1
    fi
  done
}

ehb_verify_runtime_checkout() {
  : "${PROJECT_ROOT:?PROJECT_ROOT must be set by the submission workflow}"
  : "${EXPECTED_COMMIT:?EXPECTED_COMMIT must be set by the submission workflow}"

  local actual_commit
  actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
  if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
    echo "[error] checkout commit ${actual_commit} does not match ${EXPECTED_COMMIT}" >&2
    return 1
  fi
  if [[ -n "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)" ]]; then
    echo "[error] runtime checkout is dirty: ${PROJECT_ROOT}" >&2
    return 1
  fi
}

ehb_add_production_domain_args() {
  local target_array="$1"
  local -n args_ref="${target_array}"

  args_ref+=(
    --region "${REGION}"
    --margin-n "${MARGIN_N}"
    --zg-bottom "${ZG_BOTTOM}"
    --zg-top-pa "${ZG_TOP_PA}"
  )

  if [[ "${ZG_BOTTOM}" == "pressure_level" ]]; then
    : "${ZG_BOTTOM_PA:?ZG_BOTTOM_PA must be set when ZG_BOTTOM=pressure_level}"
    args_ref+=(--zg-bottom-pa "${ZG_BOTTOM_PA}")
  fi

  if ehb_bool_enabled "${ALLOW_BOTTOM_OVERFLOW}"; then
    args_ref+=(--allow-bottom-overflow)
  else
    args_ref+=(--no-allow-bottom-overflow)
  fi

  if ehb_bool_enabled "${ENABLE_BENCHMARK_VARIABLES}"; then
    args_ref+=(--include-benchmark-variables)
  fi
}

ehb_build_production_run_budget_args() {
  local target_array="$1"
  local -n args_ref="${target_array}"

  args_ref=(
    --data-source "${DATA_SOURCE}"
    --production-output-dir "${PRODUCTION_OUTPUT_DIR}"
  )
  ehb_add_production_domain_args "${target_array}"

  if [[ "${DATA_SOURCE}" == "staged_arco_cache" ]]; then
    ehb_require_consolidated_staged_cache
    args_ref+=(--staged-cache-root "${STAGED_CACHE_ROOT}")
  fi

  if ehb_bool_enabled "${ENABLE_DIAGNOSTIC_PLOTS}"; then
    args_ref+=(--diagnostic-plots)
  else
    args_ref+=(--no-diagnostic-plots)
  fi

  if ehb_bool_enabled "${ENABLE_CONSTANT_TEMPERATURE_TEST}"; then
    args_ref+=(--constant-temperature-test)
  else
    args_ref+=(--no-constant-temperature-test)
  fi
}

ehb_build_production_staged_retrieval_args() {
  local target_array="$1"
  local time_start="$2"
  local time_end="$3"
  local shard_root="$4"
  local -n args_ref="${target_array}"

  ehb_require_staged_cache_root "production staged ARCO retrieval"
  if [[ -z "${shard_root}" ]]; then
    echo "[error] production staged ARCO retrieval requires a yearly shard root." >&2
    return 1
  fi
  args_ref=(
    --staged-cache-root "${shard_root}"
    --time-start "${time_start}"
    --time-end "${time_end}"
  )
  ehb_add_production_domain_args "${target_array}"
  args_ref+=(--stage-time-chunk "${STAGED_ARCO_TIME_CHUNK}")
  args_ref+=(--stage-attempt-timeout-seconds "${STAGED_ARCO_ATTEMPT_TIMEOUT_SECONDS}")
}

ehb_year_shard_root() {
  local year="$1"
  printf "%s/shards/year=%04d\n" "${STAGED_CACHE_ROOT}" "${year}"
}

ehb_build_staged_campaign_init_args() {
  local target_array="$1"
  local -n args_ref="${target_array}"

  args_ref=(
    init
    --cache-root "${STAGED_CACHE_ROOT}"
    --campaign-id "${CAMPAIGN_ID}"
    --start-year "${START_YEAR}"
    --end-year "${END_YEAR}"
    --start-month-day "${RUN_START_MONTH_DAY}"
    --end-month-day "${RUN_END_MONTH_DAY}"
    --region "${REGION}"
    --margin-n "${MARGIN_N}"
    --zg-top-pa "${ZG_TOP_PA}"
    --zg-bottom "${ZG_BOTTOM}"
    --time-chunk "${STAGED_ARCO_TIME_CHUNK}"
    --attempt-timeout-seconds "${STAGED_ARCO_ATTEMPT_TIMEOUT_SECONDS}"
  )
  if [[ "${ZG_BOTTOM}" == "pressure_level" ]]; then
    : "${ZG_BOTTOM_PA:?ZG_BOTTOM_PA must be set when ZG_BOTTOM=pressure_level}"
    args_ref+=(--zg-bottom-pa "${ZG_BOTTOM_PA}")
  fi
  if ehb_bool_enabled "${ALLOW_BOTTOM_OVERFLOW}"; then
    args_ref+=(--allow-bottom-overflow)
  else
    args_ref+=(--no-allow-bottom-overflow)
  fi
  if ehb_bool_enabled "${ENABLE_BENCHMARK_VARIABLES}"; then
    args_ref+=(--include-benchmark-variables)
  else
    args_ref+=(--no-include-benchmark-variables)
  fi
}

ehb_production_year_for_task() {
  local task_id="$1"
  printf "%d\n" $((START_YEAR + 10#${task_id}))
}

ehb_validate_production_year() {
  local year="$1"

  if (( year > END_YEAR )); then
    echo "[error] Computed YEAR=${year} exceeds END_YEAR=${END_YEAR}" >&2
    return 1
  fi
}

ehb_build_production_time_window() {
  local year="$1"
  local start_var="$2"
  local end_var="$3"
  local -n time_start_ref="${start_var}"
  local -n time_end_ref="${end_var}"

  time_start_ref=$(printf "%04d-%sT00:00:00" "${year}" "${RUN_START_MONTH_DAY}")
  time_end_ref=$(printf "%04d-%sT23:00:00" "${year}" "${RUN_END_MONTH_DAY}")
}
