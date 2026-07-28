from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_DIR = PROJECT_ROOT / "schedulers"
SETTINGS = SCHEDULER_DIR / "production_run_cli_settings.sh"
RUNNER = SCHEDULER_DIR / "schedule_run_budget_production.sh"
SUBMITTER = SCHEDULER_DIR / "submit_run_budget_production.sh"


def _run_bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )


def test_production_shell_scripts_have_valid_syntax():
    for path in (SETTINGS, RUNNER, SUBMITTER):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_direct_production_defaults_and_year_mapping():
    completed = _run_bash(
        f"""
        set -euo pipefail
        HOME=/tmp/ehb-test-home
        source {SETTINGS!s}
        printf '%s\\n' "$DATA_SOURCE" "$RUN_GROUP" "$RUN_ID" "$REGION" "$START_YEAR" "$END_YEAR"
        ehb_production_year_for_task 85
        """
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "arco_era5",
        "pnw",
        "pnw_hotz_surface_700hPa_1940_2025_second_attempt",
        "pnw_hotz",
        "1940",
        "2025",
        "2025",
    ]


def test_existing_nonempty_year_is_complete(tmp_path):
    annual_dir = tmp_path / "annual"
    annual_dir.mkdir()
    (annual_dir / "heat_budget_1941.nc").write_bytes(b"netcdf")
    completed = _run_bash(
        f"""
        set -euo pipefail
        HOME=/tmp/ehb-test-home
        PRODUCTION_OUTPUT_DIR={tmp_path!s}
        source {SETTINGS!s}
        ehb_year_is_complete 1941
        ! ehb_year_is_complete 1942
        """
    )

    assert completed.returncode == 0, completed.stderr


def test_direct_production_rejects_staged_cache_source():
    completed = _run_bash(
        f"""
        set -euo pipefail
        HOME=/tmp/ehb-test-home
        DATA_SOURCE=staged_arco_cache
        source {SETTINGS!s}
        ehb_require_direct_arco_source
        """
    )

    assert completed.returncode != 0
    assert "requires DATA_SOURCE=arco_era5" in completed.stderr


def test_production_paths_must_be_outside_checkout():
    completed = _run_bash(
        f"""
        set -euo pipefail
        HOME=/tmp/ehb-test-home
        PROJECT_ROOT=/tmp/ehb-test-home/eulerian-heat-budget/production/checkout
        PRODUCTION_OUTPUT_DIR="$PROJECT_ROOT/results"
        source {SETTINGS!s}
        ehb_require_external_production_paths
        """
    )

    assert completed.returncode != 0
    assert "must be outside the Git checkout" in completed.stderr


def test_scheduler_has_no_legacy_checkout_path():
    legacy_path = "/home/mhpereir/eulerian_heat_budget"

    assert legacy_path not in RUNNER.read_text()
    assert legacy_path not in SUBMITTER.read_text()
