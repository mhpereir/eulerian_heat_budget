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
CAMPAIGN_ID="${CAMPAIGN_ID:-alaska-surface-700hpa-1940-2025}"
RUN_ID="${RUN_ID:-${CAMPAIGN_ID}}"
REGION="${REGION:-alaska}"
MARGIN_N="${MARGIN_N:-1}"
ZG_TOP_PA="${ZG_TOP_PA:-70000}"
ZG_BOTTOM="${ZG_BOTTOM:-surface_pressure}"
ZG_BOTTOM_PA="${ZG_BOTTOM_PA:-}"
ALLOW_BOTTOM_OVERFLOW="${ALLOW_BOTTOM_OVERFLOW:-1}"

EHB_WORKSPACE_ROOT="${EHB_WORKSPACE_ROOT:-${HOME:?HOME must be set}/eulerian-heat-budget}"
EHB_CAMPAIGN_DATA_ROOT="${EHB_CAMPAIGN_DATA_ROOT:-${EHB_WORKSPACE_ROOT}/campaign-data}"
EHB_STAGED_ZARR_ROOT="${EHB_STAGED_ZARR_ROOT:-${EHB_CAMPAIGN_DATA_ROOT}/staged-zarr}"
EHB_RUN_BUDGET_ROOT="${EHB_RUN_BUDGET_ROOT:-${EHB_CAMPAIGN_DATA_ROOT}/run-budget}"
EHB_LOG_ROOT="${EHB_LOG_ROOT:-${EHB_CAMPAIGN_DATA_ROOT}/logs}"
EHB_OUTPUT_ROOT="${EHB_OUTPUT_ROOT:-${EHB_RUN_BUDGET_ROOT}}"

STAGED_CACHE_BASE_ROOT="${STAGED_CACHE_BASE_ROOT:-${EHB_STAGED_ZARR_ROOT}/${REGION}}"
STAGED_CACHE_ROOT="${STAGED_CACHE_ROOT:-${STAGED_CACHE_BASE_ROOT}/${CAMPAIGN_ID}}"
STAGED_RUN_MANIFEST_PATH="${STAGED_RUN_MANIFEST_PATH:-${STAGED_CACHE_ROOT}/production_run.json}"
PRODUCTION_OUTPUT_DIR="${PRODUCTION_OUTPUT_DIR:-${EHB_RUN_BUDGET_ROOT}/${REGION}/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${EHB_LOG_ROOT}/${REGION}/${RUN_ID}}"
STAGED_ARCO_TIME_CHUNK="${STAGED_ARCO_TIME_CHUNK:-month}"
STAGED_ARCO_ATTEMPT_TIMEOUT_SECONDS="${STAGED_ARCO_ATTEMPT_TIMEOUT_SECONDS:-20800}"
RUN_BUDGET_WALLTIME="${RUN_BUDGET_WALLTIME:-02:00:00}"

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

ehb_require_external_production_paths() {
  : "${PROJECT_ROOT:?PROJECT_ROOT must be set by the submission workflow}"

  local project_root
  project_root=$(readlink -m -- "${PROJECT_ROOT}")
  local label
  local path
  for label in STAGED_CACHE_ROOT STAGED_RUN_MANIFEST_PATH PRODUCTION_OUTPUT_DIR LOG_DIR; do
    path=$(readlink -m -- "${!label}")
    case "${path}/" in
      "${project_root}/"*)
        echo "[error] ${label} must be outside the Git checkout: ${path}" >&2
        return 1
        ;;
    esac
  done

  local expected_manifest_path
  expected_manifest_path=$(readlink -m -- "${STAGED_CACHE_ROOT}/production_run.json")
  if [[ "$(readlink -m -- "${STAGED_RUN_MANIFEST_PATH}")" != "${expected_manifest_path}" ]]; then
    echo "[error] STAGED_RUN_MANIFEST_PATH must be ${expected_manifest_path}." >&2
    return 1
  fi
}

ehb_require_venus_production_checkout() {
  : "${PROJECT_ROOT:?PROJECT_ROOT must be set by the submission workflow}"

  local required_branch="production_development_staged"
  local project_root
  local production_root
  local git_root
  local current_branch
  local upstream_branch

  project_root=$(readlink -m -- "${PROJECT_ROOT}")
  production_root=$(readlink -m -- "${EHB_WORKSPACE_ROOT}/production")
  case "${project_root}" in
    "${production_root}/"*) ;;
    *)
      echo "[error] Venus production checkout must be below ${production_root}: ${project_root}" >&2
      return 1
      ;;
  esac

  if ! git_root=$(git -C "${project_root}" rev-parse --show-toplevel); then
    echo "[error] PROJECT_ROOT is not a Git checkout: ${project_root}" >&2
    return 1
  fi
  git_root=$(readlink -m -- "${git_root}")
  if [[ "${git_root}" != "${project_root}" ]]; then
    echo "[error] PROJECT_ROOT is not the Git root: ${project_root}" >&2
    return 1
  fi

  if ! current_branch=$(git -C "${project_root}" symbolic-ref --quiet --short HEAD); then
    echo "[error] Venus production checkout must use the named ${required_branch} branch." >&2
    return 1
  fi
  if [[ "${current_branch}" != "${required_branch}" ]]; then
    echo "[error] Venus production requires branch ${required_branch}, not ${current_branch}." >&2
    return 1
  fi

  if ! upstream_branch=$(
    git -C "${project_root}" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'
  ); then
    echo "[error] ${required_branch} must track origin/${required_branch}." >&2
    return 1
  fi
  if [[ "${upstream_branch}" != "origin/${required_branch}" ]]; then
    echo "[error] Venus production upstream must be origin/${required_branch}, not ${upstream_branch}." >&2
    return 1
  fi
}

ehb_verify_production_submission_checkout() {
  ehb_require_venus_production_checkout || return 1

  local project_root
  local required_ref="refs/heads/production_development_staged"
  local actual_commit
  local remote_commit
  local remote_ref
  local status_output

  project_root=$(readlink -m -- "${PROJECT_ROOT}")
  if ! status_output=$(
    git -C "${project_root}" status --porcelain --untracked-files=normal
  ); then
    echo "[error] Could not inspect production checkout status: ${project_root}" >&2
    return 1
  fi
  if [[ -n "${status_output}" ]]; then
    echo "[error] Refusing to submit from a dirty checkout: ${project_root}" >&2
    return 1
  fi

  if ! actual_commit=$(git -C "${project_root}" rev-parse HEAD); then
    echo "[error] Could not resolve production checkout HEAD: ${project_root}" >&2
    return 1
  fi
  if ! read -r remote_commit remote_ref < <(
    git -C "${project_root}" ls-remote --exit-code origin "${required_ref}"
  ); then
    echo "[error] Could not resolve the live origin/production_development_staged tip." >&2
    return 1
  fi
  if [[ "${remote_ref}" != "${required_ref}" || -z "${remote_commit}" ]]; then
    echo "[error] origin did not return exactly ${required_ref}." >&2
    return 1
  fi
  if [[ "${actual_commit}" != "${remote_commit}" ]]; then
    echo "[error] Venus production checkout ${actual_commit} is not the authoritative remote tip ${remote_commit}." >&2
    echo "[error] Integrate and push all intended commits to production_development_staged before submission." >&2
    return 1
  fi

  printf '%s\n' "${actual_commit}"
}

ehb_require_staged_cache_root() {
  local context="$1"

  if [[ -z "${STAGED_CACHE_ROOT:-}" ]]; then
    echo "[error] ${context} requires STAGED_CACHE_ROOT to point to a local indexed cache root." >&2
    echo "[error] Populate it first with scripts/staged_arco_retrieval.py from an internet-capable session." >&2
    return 1
  fi
}

ehb_require_staged_run_manifest() {
  if [[ ! -f "${STAGED_RUN_MANIFEST_PATH}" ]]; then
    echo "[error] staged production manifest is missing: ${STAGED_RUN_MANIFEST_PATH}" >&2
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

  ehb_require_venus_production_checkout || return 1

  local actual_commit
  local status_output
  if ! actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD); then
    echo "[error] Could not resolve runtime checkout HEAD: ${PROJECT_ROOT}" >&2
    return 1
  fi
  if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
    echo "[error] checkout commit ${actual_commit} does not match ${EXPECTED_COMMIT}" >&2
    return 1
  fi
  if ! status_output=$(
    git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal
  ); then
    echo "[error] Could not inspect runtime checkout status: ${PROJECT_ROOT}" >&2
    return 1
  fi
  if [[ -n "${status_output}" ]]; then
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

ehb_resolve_yearly_task_index() {
  local array_index="${PBS_ARRAY_INDEX:-}"
  local serial_index="${EHB_SERIAL_TASK_INDEX:-}"
  local resolved_index

  if [[ -n "${array_index}" && -n "${serial_index}" && "${array_index}" != "${serial_index}" ]]; then
    echo "[error] PBS_ARRAY_INDEX and EHB_SERIAL_TASK_INDEX disagree." >&2
    return 1
  fi
  if [[ -n "${array_index}" ]]; then
    resolved_index="${array_index}"
  elif [[ -n "${serial_index}" ]]; then
    resolved_index="${serial_index}"
  else
    echo "[error] A yearly task requires PBS_ARRAY_INDEX or EHB_SERIAL_TASK_INDEX." >&2
    return 1
  fi

  if [[ ! "${resolved_index}" =~ ^[0-9]+$ ]]; then
    echo "[error] Yearly task index must be a nonnegative integer: ${resolved_index}" >&2
    return 1
  fi
  printf '%s\n' "${resolved_index}"
}

ehb_yearly_output_path() {
  local year="$1"
  printf "%s/annual/heat_budget_%04d.nc\n" "${PRODUCTION_OUTPUT_DIR}" "${year}"
}

ehb_year_is_complete() {
  local year="$1"
  local output_path
  output_path=$(ehb_yearly_output_path "${year}")
  [[ -s "${output_path}" ]]
}

ehb_validate_production_year() {
  local year="$1"

  if (( year < START_YEAR || year > END_YEAR )); then
    echo "[error] Computed YEAR=${year} is outside ${START_YEAR}-${END_YEAR}." >&2
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
