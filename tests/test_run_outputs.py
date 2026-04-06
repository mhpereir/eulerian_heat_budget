import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datetime import datetime
import json
from pathlib import Path
import subprocess

import pytest
import xarray as xr

from src.run_outputs import (
    GitProvenance,
    prepare_production_paths,
    prepare_run_paths,
    ProductionPaths,
    require_output_path,
    require_production_manifest,
    resolve_production_year,
    resolve_git_provenance,
    write_budget_result,
    write_production_manifest,
    write_run_info,
)
from src.specs import DataSourceConfig, DomainRequest, DomainSpec, SurfaceBehaviour


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "schedulers").mkdir()

    (repo / "src" / "tracked.py").write_text("VALUE = 1\n")
    (repo / "scripts" / "tracked.sh").write_text("#!/bin/bash\n")
    (repo / "schedulers" / "tracked.txt").write_text("tracked\n")

    _git(repo, "init")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "add", "src", "scripts", "schedulers")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "branch", "-m", "test-branch")
    return repo


def test_prepare_run_paths_uses_pbs_jobid(tmp_path):
    paths = prepare_run_paths(
        str(tmp_path),
        env={"PBS_JOBID": "2586030.venus"},
        now=datetime(2026, 3, 17, 12, 0, 0),
        pid=99,
    )

    assert paths.run_id == "2586030.venus"
    assert Path(paths.run_root) == tmp_path / "2586030.venus"
    assert Path(paths.plot_dir) == tmp_path / "2586030.venus" / "plots"
    assert Path(paths.plot_dir).is_dir()
    assert Path(paths.metadata_path) == tmp_path / "2586030.venus" / "run_info.json"


def test_write_run_info_serializes_specs_to_json(tmp_path):
    paths = prepare_run_paths(
        str(tmp_path),
        env={"PBS_JOBID": "2586030.venus"},
        now=datetime(2026, 3, 17, 12, 0, 0),
        pid=99,
    )

    request = DomainRequest(
        bbox=(40.0, 60.0, -130.0, -110.0),
        margin_n=1,
        zg_top_pressure=60000.0,
        zg_bottom="surface_pressure",
        zg_bottom_pressure=None,
    )
    source_spec = DataSourceConfig(
        kind="arco_era5",
        arco_path="gs://example-dataset.zarr",
        time_start="1961-06-01T00:00:00",
        time_end="1961-06-07T00:00:00",
    )
    domain_spec = DomainSpec(
        lat_min=40.25,
        lat_max=59.75,
        lon_min=-129.75,
        lon_max=-110.25,
        zg_top_pressure=60000.0,
        zg_bottom="surface_pressure",
        zg_bottom_pressure=None,
    )
    surface_behaviour = SurfaceBehaviour(
        allow_bottom_overflow=False,
        use_surface_variables=False,
        surface_variable_mode="combined",
    )
    git_provenance = GitProvenance(
        branch="test-branch",
        commit="1234567890abcdef1234567890abcdef12345678",
        dirty=True,
    )

    metadata_path = write_run_info(
        paths,
        request=request,
        source_spec=source_spec,
        domain_spec=domain_spec,
        surface_behaviour=surface_behaviour,
        git_provenance=git_provenance,
        cli_args={"lat_min": 40.0, "in_surface_variables": False},
        env={"PBS_JOBID": "2586030.venus"},
        now=datetime(2026, 3, 17, 12, 30, 0),
    )

    payload = json.loads(Path(metadata_path).read_text())

    assert payload["run_id"] == "2586030.venus"
    assert payload["pbs_job_id"] == "2586030.venus"
    assert payload["plot_dir"] == str(tmp_path / "2586030.venus" / "plots")
    assert payload["request"]["bbox"] == [40.0, 60.0, -130.0, -110.0]
    assert payload["source_spec"]["kind"] == "arco_era5"
    assert payload["domain_spec"]["lat_min"] == 40.25
    assert payload["surface_behaviour"]["surface_variable_mode"] == "combined"
    assert payload["git"]["branch"] == "test-branch"
    assert payload["git"]["commit"] == "1234567890abcdef1234567890abcdef12345678"
    assert payload["git"]["dirty"] is True
    assert payload["cli_args"]["in_surface_variables"] is False


def test_prepare_production_paths_creates_shared_layout(tmp_path):
    paths = prepare_production_paths(str(tmp_path / "production"), year=1940)

    assert Path(paths.root_dir) == tmp_path / "production"
    assert Path(paths.annual_dir) == tmp_path / "production" / "annual"
    assert Path(paths.plot_root) == tmp_path / "production" / "plots"
    assert Path(paths.plot_dir) == tmp_path / "production" / "plots" / "1940"
    assert Path(paths.output_path) == tmp_path / "production" / "annual" / "heat_budget_1940.nc"
    assert Path(paths.manifest_path) == tmp_path / "production" / "production_run.json"
    assert Path(paths.annual_dir).is_dir()
    assert Path(paths.plot_dir).is_dir()


def test_resolve_production_year_rejects_cross_year_ranges():
    with pytest.raises(ValueError, match="same calendar year"):
        resolve_production_year(
            time_start="1940-12-31T00:00:00",
            time_end="1941-01-01T00:00:00",
        )


def test_write_production_manifest_serializes_shared_campaign_metadata(tmp_path):
    paths = prepare_production_paths(str(tmp_path / "production"))

    request = DomainRequest(
        bbox=(40.0, 60.0, -130.0, -110.0),
        margin_n=1,
        zg_top_pressure=60000.0,
        zg_bottom="surface_pressure",
        zg_bottom_pressure=None,
    )
    source_spec = DataSourceConfig(
        kind="arco_era5",
        arco_path="gs://example-dataset.zarr",
        time_start=None,
        time_end=None,
    )
    surface_behaviour = SurfaceBehaviour(
        allow_bottom_overflow=False,
        use_surface_variables=False,
        surface_variable_mode="none",
    )
    git_provenance = GitProvenance(
        branch="test-branch",
        commit="1234567890abcdef1234567890abcdef12345678",
        dirty=False,
    )

    manifest_path = write_production_manifest(
        paths,
        production_start_year=1940,
        production_end_year=2025,
        request=request,
        source_spec=source_spec,
        surface_behaviour=surface_behaviour,
        git_provenance=git_provenance,
        cli_args={"production_output_dir": str(tmp_path / "production")},
        now=datetime(2026, 4, 6, 10, 0, 0),
    )

    payload = json.loads(Path(manifest_path).read_text())

    assert payload["production_start_year"] == 1940
    assert payload["production_end_year"] == 2025
    assert payload["root_dir"] == str(tmp_path / "production")
    assert payload["annual_dir"] == str(tmp_path / "production" / "annual")
    assert payload["plot_root"] == str(tmp_path / "production" / "plots")
    assert payload["source_spec"]["time_start"] is None
    assert payload["source_spec"]["time_end"] is None
    assert payload["git"]["dirty"] is False


def test_require_production_manifest_raises_when_missing(tmp_path):
    paths = prepare_production_paths(str(tmp_path / "production"))

    with pytest.raises(FileNotFoundError, match="Production manifest not found"):
        require_production_manifest(paths)


def test_require_output_path_fails_when_existing_without_overwrite(tmp_path):
    output_path = tmp_path / "production" / "annual" / "heat_budget_1940.nc"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("existing\n")

    with pytest.raises(FileExistsError, match="Output already exists"):
        require_output_path(str(output_path), overwrite=False)


def test_require_output_path_overwrites_existing_file_when_requested(tmp_path):
    output_path = tmp_path / "production" / "annual" / "heat_budget_1940.nc"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("existing\n")

    resolved = require_output_path(str(output_path), overwrite=True)

    assert resolved == str(output_path)
    assert not output_path.exists()


def test_write_budget_result_writes_netcdf_output(tmp_path):
    output_path = tmp_path / "production" / "annual" / "heat_budget_1940.nc"
    ds_budget = xr.Dataset({"value": xr.DataArray([1.0, 2.0], dims=("time",))})

    written = write_budget_result(ds_budget, str(output_path), overwrite=False)

    assert written == str(output_path)
    assert output_path.is_file()


def test_resolve_git_provenance_returns_branch_commit_and_clean_status(tmp_path):
    repo = _make_repo(tmp_path)

    provenance = resolve_git_provenance(repo)

    assert provenance.branch == "test-branch"
    assert provenance.commit == _git(repo, "rev-parse", "HEAD")
    assert provenance.dirty is False


def test_resolve_git_provenance_marks_tracked_runtime_changes_dirty(tmp_path):
    repo = _make_repo(tmp_path)
    tracked_file = repo / "src" / "tracked.py"
    tracked_file.write_text("VALUE = 2\n")

    provenance = resolve_git_provenance(repo)

    assert provenance.dirty is True


def test_resolve_git_provenance_ignores_generated_noise(tmp_path):
    repo = _make_repo(tmp_path)
    pycache_dir = repo / "src" / "__pycache__"
    pycache_dir.mkdir()
    cached_file = pycache_dir / "tracked.cpython-312.pyc"
    cached_file.write_text("compiled-v1\n")
    _git(repo, "add", str(cached_file.relative_to(repo)))
    _git(repo, "commit", "-m", "add generated artifact")

    cached_file.write_text("compiled-v2\n")

    provenance = resolve_git_provenance(repo)

    assert provenance.dirty is False


def test_resolve_git_provenance_raises_outside_git_repo(tmp_path):
    with pytest.raises(ValueError, match="not a git repository"):
        resolve_git_provenance(tmp_path)


def test_resolve_git_provenance_raises_on_detached_head(tmp_path):
    repo = _make_repo(tmp_path)
    _git(repo, "checkout", "--detach")

    with pytest.raises(ValueError, match="detached HEAD"):
        resolve_git_provenance(repo)
