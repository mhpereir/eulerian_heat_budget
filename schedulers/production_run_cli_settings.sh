# Shared settings for direct ARCO ERA5 production heat-budget runs on Venus.

START_YEAR="${START_YEAR:-1940}"
END_YEAR="${END_YEAR:-2025}"
RUN_START_MONTH_DAY="${RUN_START_MONTH_DAY:-05-01}"
RUN_END_MONTH_DAY="${RUN_END_MONTH_DAY:-10-31}"

DATA_SOURCE="${DATA_SOURCE:-arco_era5}"
RUN_ID="${RUN_ID:-pnw_hotz_surface_700hPa_1940_2025_second_attempt}"
RUN_GROUP="${RUN_GROUP:-pnw}"
REGION="${REGION:-pnw_hotz}"
MARGIN_N="${MARGIN_N:-1}"
ZG_TOP_PA="${ZG_TOP_PA:-70000}"
ZG_BOTTOM="${ZG_BOTTOM:-surface_pressure}"
ZG_BOTTOM_PA="${ZG_BOTTOM_PA:-}"
ALLOW_BOTTOM_OVERFLOW="${ALLOW_BOTTOM_OVERFLOW:-1}"

EHB_WORKSPACE_ROOT="${EHB_WORKSPACE_ROOT:-${HOME:?HOME must be set}/eulerian-heat-budget}"
EHB_CAMPAIGN_DATA_ROOT="${EHB_CAMPAIGN_DATA_ROOT:-${EHB_WORKSPACE_ROOT}/campaign-data}"
EHB_RUN_BUDGET_ROOT="${EHB_RUN_BUDGET_ROOT:-${EHB_CAMPAIGN_DATA_ROOT}/run-budget}"
EHB_LOG_ROOT="${EHB_LOG_ROOT:-${EHB_CAMPAIGN_DATA_ROOT}/logs}"
PRODUCTION_OUTPUT_DIR="${PRODUCTION_OUTPUT_DIR:-${EHB_RUN_BUDGET_ROOT}/${RUN_GROUP}/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${EHB_LOG_ROOT}/${RUN_GROUP}/${RUN_ID}}"

INIT_MANIFEST_ONLY="${INIT_MANIFEST_ONLY:-0}"
ENABLE_DIAGNOSTIC_PLOTS="${ENABLE_DIAGNOSTIC_PLOTS:-1}"
ENABLE_CONSTANT_TEMPERATURE_TEST="${ENABLE_CONSTANT_TEMPERATURE_TEST:-0}"
ENABLE_BENCHMARK_VARIABLES="${ENABLE_BENCHMARK_VARIABLES:-0}"
MANIFEST_PATH="${MANIFEST_PATH:-${PRODUCTION_OUTPUT_DIR}/production_run.json}"
MANIFEST_LOCK_DIR="${MANIFEST_LOCK_DIR:-${PRODUCTION_OUTPUT_DIR}/.manifest_init.lock}"
MANIFEST_WAIT_SECONDS="${MANIFEST_WAIT_SECONDS:-300}"

ehb_bool_enabled() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

ehb_require_direct_arco_source() {
  if [[ "${DATA_SOURCE}" != "arco_era5" ]]; then
    echo "[error] Direct production requires DATA_SOURCE=arco_era5, not ${DATA_SOURCE}." >&2
    return 1
  fi
}

ehb_require_external_production_paths() {
  : "${PROJECT_ROOT:?PROJECT_ROOT must be set by the submission workflow}"

  local project_root
  project_root=$(readlink -m -- "${PROJECT_ROOT}")
  local label
  local path
  for label in PRODUCTION_OUTPUT_DIR LOG_DIR; do
    path=$(readlink -m -- "${!label}")
    case "${path}/" in
      "${project_root}/"*)
        echo "[error] ${label} must be outside the Git checkout: ${path}" >&2
        return 1
        ;;
    esac
  done
}

ehb_require_venus_production_checkout() {
  : "${PROJECT_ROOT:?PROJECT_ROOT must be set by the submission workflow}"

  local required_branch="production_development"
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

  git_root=$(git -C "${project_root}" rev-parse --show-toplevel)
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

  upstream_branch=$(
    git -C "${project_root}" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'
  )
  if [[ "${upstream_branch}" != "origin/${required_branch}" ]]; then
    echo "[error] Venus production upstream must be origin/${required_branch}, not ${upstream_branch}." >&2
    return 1
  fi
}

ehb_verify_production_submission_checkout() {
  ehb_require_venus_production_checkout || return 1

  local required_ref="refs/heads/production_development"
  local actual_commit
  local remote_commit
  local remote_ref
  local status_output

  status_output=$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)
  if [[ -n "${status_output}" ]]; then
    echo "[error] Refusing to submit from a dirty checkout: ${PROJECT_ROOT}" >&2
    return 1
  fi

  actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
  if ! read -r remote_commit remote_ref < <(
    git -C "${PROJECT_ROOT}" ls-remote --exit-code origin "${required_ref}"
  ); then
    echo "[error] Could not resolve the live origin/production_development tip." >&2
    return 1
  fi
  if [[ "${remote_ref}" != "${required_ref}" || -z "${remote_commit}" ]]; then
    echo "[error] origin did not return exactly ${required_ref}." >&2
    return 1
  fi
  if [[ "${actual_commit}" != "${remote_commit}" ]]; then
    echo "[error] Checkout ${actual_commit} is not the authoritative remote tip ${remote_commit}." >&2
    return 1
  fi

  printf '%s\n' "${actual_commit}"
}

ehb_verify_runtime_checkout() {
  : "${EXPECTED_COMMIT:?EXPECTED_COMMIT must be set by the submission workflow}"
  ehb_require_venus_production_checkout || return 1

  local actual_commit
  local status_output
  actual_commit=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
  if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
    echo "[error] Checkout commit ${actual_commit} does not match ${EXPECTED_COMMIT}." >&2
    return 1
  fi
  status_output=$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)
  if [[ -n "${status_output}" ]]; then
    echo "[error] Runtime checkout is dirty: ${PROJECT_ROOT}" >&2
    return 1
  fi
}

ehb_build_production_run_budget_args() {
  local target_array="$1"
  local -n args_ref="${target_array}"

  ehb_require_direct_arco_source
  args_ref=(
    --data-source "${DATA_SOURCE}"
    --production-output-dir "${PRODUCTION_OUTPUT_DIR}"
    --region "${REGION}"
    --margin-n "${MARGIN_N}"
    --zg-top-pa "${ZG_TOP_PA}"
    --zg-bottom "${ZG_BOTTOM}"
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
  if ehb_bool_enabled "${ENABLE_BENCHMARK_VARIABLES}"; then
    args_ref+=(--include-benchmark-variables)
  fi
}

ehb_production_year_for_task() {
  local task_id="$1"
  printf "%d\n" $((START_YEAR + 10#${task_id}))
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
