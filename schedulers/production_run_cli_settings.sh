# Shared CLI settings for production ARCO staging and production budget runs.
#
# This file is sourced by:
# - schedulers/schedule_run_budget_production_slurm.sh
# - schedulers/schedule_staged_arco_retrieval_production_slurm.sh
#
# Keep production staging and compute settings here so they cannot drift.

START_YEAR="${START_YEAR:-1946}"
END_YEAR="${END_YEAR:-2025}"
RUN_START_MONTH_DAY="${RUN_START_MONTH_DAY:-05-01}"
RUN_END_MONTH_DAY="${RUN_END_MONTH_DAY:-10-31}"

DATA_SOURCE="${DATA_SOURCE:-staged_arco_cache}"
REGION="${REGION:-eastern_canada}"
ZG_TOP_PA="${ZG_TOP_PA:-70000}"
ZG_BOTTOM="${ZG_BOTTOM:-surface_pressure}"
ZG_BOTTOM_PA="${ZG_BOTTOM_PA:-}"
ALLOW_BOTTOM_OVERFLOW="${ALLOW_BOTTOM_OVERFLOW:-1}"

INIT_MANIFEST_ONLY="${INIT_MANIFEST_ONLY:-0}"
ENABLE_DIAGNOSTIC_PLOTS="${ENABLE_DIAGNOSTIC_PLOTS:-1}"
ENABLE_CONSTANT_TEMPERATURE_TEST="${ENABLE_CONSTANT_TEMPERATURE_TEST:-0}"
ENABLE_BENCHMARK_VARIABLES="${ENABLE_BENCHMARK_VARIABLES:-0}"
MANIFEST_WAIT_SECONDS="${MANIFEST_WAIT_SECONDS:-300}"

ehb_bool_enabled() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

ehb_default_production_output_dir() {
  if [[ -n "${SCRATCH:-}" ]]; then
    printf "%s\n" "${SCRATCH}/eulerian_heat_budget/results/staged_arco_cache/production"
  else
    printf "%s\n" "${REPO_ROOT:-$(pwd)}/results/staged_arco_cache/production"
  fi
}

ehb_default_staged_cache_root() {
  if [[ -n "${SCRATCH:-}" ]]; then
    printf "%s\n" "${SCRATCH}/eulerian_heat_budget/results/staged_arco_cache/production"
  else
    printf "%s\n" "${REPO_ROOT:-$(pwd)}/results/staged_arco_cache/production"
  fi
}

DEFAULT_PRODUCTION_OUTPUT_DIR="${DEFAULT_PRODUCTION_OUTPUT_DIR:-$(ehb_default_production_output_dir)}"
PRODUCTION_OUTPUT_DIR="${PRODUCTION_OUTPUT_DIR:-${DEFAULT_PRODUCTION_OUTPUT_DIR}}"
MANIFEST_PATH="${MANIFEST_PATH:-${PRODUCTION_OUTPUT_DIR}/production_run.json}"
MANIFEST_LOCK_DIR="${MANIFEST_LOCK_DIR:-${PRODUCTION_OUTPUT_DIR}/.manifest_init.lock}"

DEFAULT_STAGED_CACHE_ROOT="${DEFAULT_STAGED_CACHE_ROOT:-$(ehb_default_staged_cache_root)}"
STAGED_CACHE_ROOT="${STAGED_CACHE_ROOT-${DEFAULT_STAGED_CACHE_ROOT}}"
STAGED_ARCO_TIME_CHUNK="${STAGED_ARCO_TIME_CHUNK:-month}"

ehb_require_staged_cache_root() {
  local context="$1"

  if [[ -z "${STAGED_CACHE_ROOT:-}" ]]; then
    echo "[error] ${context} requires STAGED_CACHE_ROOT to point to a local indexed cache root." >&2
    echo "[error] Populate it first with scripts/staged_arco_retrieval.py from an internet-capable session." >&2
    return 1
  fi
}

ehb_add_production_domain_args() {
  local target_array="$1"
  local -n args_ref="${target_array}"

  args_ref+=(
    --region "${REGION}"
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
    ehb_require_staged_cache_root "DATA_SOURCE=staged_arco_cache"
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
  local -n args_ref="${target_array}"

  ehb_require_staged_cache_root "production staged ARCO retrieval"
  args_ref=(
    --staged-cache-root "${STAGED_CACHE_ROOT}"
    --time-start "${time_start}"
    --time-end "${time_end}"
  )
  ehb_add_production_domain_args "${target_array}"
  args_ref+=(--stage-time-chunk "${STAGED_ARCO_TIME_CHUNK}")
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
