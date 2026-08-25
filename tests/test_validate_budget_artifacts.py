import importlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import xarray as xr


PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

validator = importlib.import_module("scripts.validate_budget_artifacts")


def _dataset(profile_name):
    profile = validator.PROFILES[profile_name]
    time = np.arange(
        np.datetime64(profile.time_start),
        np.datetime64(profile.time_end) + np.timedelta64(1, "h"),
        np.timedelta64(1, "h"),
    )
    names = list(validator.CORE_REQUIRED)
    if profile.benchmark_variables:
        names.extend(validator.BENCHMARK_REQUIRED)
    else:
        names.extend(("flux_contribution_bottom", "mass_flux_contribution_bottom"))
    ds = xr.Dataset(
        {name: ("time", np.zeros(time.size, dtype=np.float64)) for name in names},
        coords={"time": time},
    )
    ds["domain_volume"][:] = 1.0
    ds["domain_volume_true"][:] = 1.0
    ds["T_domain_avg"][:] = 250.0
    ds["T_scale"] = xr.DataArray(10.0)
    return ds


def _run_info(profile_name, commit, cache):
    profile = validator.PROFILES[profile_name]
    return {
        "run_id": "test.venus",
        "request": {
            "bbox": [40, 60, -130, -110],
            "margin_n": 1,
            "zg_top_pressure": profile.zg_top_pressure,
            "zg_bottom": profile.zg_bottom,
            "zg_bottom_pressure": profile.zg_bottom_pressure,
        },
        "source_spec": {
            "kind": "staged_arco_cache",
            "time_start": profile.input_start,
            "time_end": profile.input_end,
            "staged_cache_root": str(cache),
        },
        "surface_behaviour": {
            "allow_bottom_overflow": profile.allow_bottom_overflow,
            "use_surface_variables": False,
            "surface_variable_mode": "none",
        },
        "git": {"branch": "fix/test", "commit": commit, "dirty": False},
        "cli_args": {
            "region": "pnw_bartusek",
            "include_benchmark_variables": profile.benchmark_variables,
            "diagnostic_plots": True,
            "constant_temperature_test": False,
            "write_netcdf": True,
        },
    }


def _write_run(tmp_path, profile_name, *, commit="a" * 40):
    run_dir = tmp_path / profile_name
    plot_dir = run_dir / "plots" / "diagnostics"
    plot_dir.mkdir(parents=True)
    cache = tmp_path / "cache"
    _dataset(profile_name).to_netcdf(run_dir / "heat_budget.nc", engine="h5netcdf")
    (run_dir / "run_info.json").write_text(
        json.dumps(_run_info(profile_name, commit, cache)),
        encoding="utf-8",
    )
    profile = validator.PROFILES[profile_name]
    names = list(profile.required_pngs)
    names.extend(f"extra-{index}.png" for index in range(profile.png_count - len(names)))
    for index, name in enumerate(names):
        Image.fromarray(np.full((3, 4, 3), index, dtype=np.uint8)).save(plot_dir / name)
    return run_dir, cache, commit


@pytest.mark.parametrize("profile_name", sorted(validator.PROFILES))
def test_validate_run_accepts_complete_profile(tmp_path, profile_name):
    run_dir, cache, commit = _write_run(tmp_path, profile_name)

    report = validator.validate_run(
        run_dir,
        profile_name=profile_name,
        expected_commit=commit,
        expected_cache=cache,
    )

    assert report["passed"] is True
    assert (
        report["scientific"]["time_count"]
        == validator.PROFILES[profile_name].time_count
    )
    assert report["png_count"] == validator.PROFILES[profile_name].png_count


def test_validate_run_compares_dataset_and_required_plot_pixels(tmp_path):
    candidate, cache, commit = _write_run(tmp_path / "candidate", "fixed-500-300-2021")
    reference, _, _ = _write_run(tmp_path / "reference", "fixed-500-300-2021")

    report = validator.validate_run(
        candidate,
        profile_name="fixed-500-300-2021",
        expected_commit=commit,
        expected_cache=cache,
        reference_dir=reference,
    )

    assert report["reference_comparison"]["dataset_identical"] is True
    assert report["reference_comparison"]["dataset_scientifically_equivalent"] is True
    assert len(report["reference_comparison"]["pixel_identical_plots"]) == 2
    assert report["reference_comparison"]["plot_comparison"] == {
        "performed": True,
        "reason": None,
    }


def test_validate_run_compares_dataset_when_reference_has_no_plots(tmp_path):
    candidate, cache, commit = _write_run(tmp_path / "candidate", "fixed-500-300-2021")
    reference, _, _ = _write_run(tmp_path / "reference", "fixed-500-300-2021")
    for path in reference.rglob("*.png"):
        path.unlink()

    report = validator.validate_run(
        candidate,
        profile_name="fixed-500-300-2021",
        expected_commit=commit,
        expected_cache=cache,
        reference_dir=reference,
    )

    comparison = report["reference_comparison"]
    assert comparison["dataset_identical"] is True
    assert comparison["dataset_scientifically_equivalent"] is True
    assert comparison["pixel_identical_plots"] == []
    assert comparison["plot_comparison"] == {
        "performed": False,
        "reason": "Reference run contains no PNG plots.",
    }


def test_validate_run_accepts_roundoff_equivalent_reference(tmp_path):
    candidate, cache, commit = _write_run(tmp_path / "candidate", "fixed-500-300-2021")
    reference, _, _ = _write_run(tmp_path / "reference", "fixed-500-300-2021")
    with xr.open_dataset(candidate / "heat_budget.nc", engine="h5netcdf") as opened:
        changed = opened.load()
    changed["T_domain_avg"][0] += 1.0e-9
    changed.to_netcdf(candidate / "heat_budget.nc", engine="h5netcdf", mode="w")

    report = validator.validate_run(
        candidate,
        profile_name="fixed-500-300-2021",
        expected_commit=commit,
        expected_cache=cache,
        reference_dir=reference,
    )

    comparison = report["reference_comparison"]
    assert comparison["dataset_identical"] is False
    assert comparison["dataset_scientifically_equivalent"] is True
    assert comparison["variables"]["T_domain_avg"]["identical"] is False


def test_validate_run_rejects_difference_above_scientific_tolerance(tmp_path):
    candidate, cache, commit = _write_run(tmp_path / "candidate", "fixed-500-300-2021")
    reference, _, _ = _write_run(tmp_path / "reference", "fixed-500-300-2021")
    with xr.open_dataset(candidate / "heat_budget.nc", engine="h5netcdf") as opened:
        changed = opened.load()
    changed["T_domain_avg"][0] += 1.0e-4
    changed.to_netcdf(candidate / "heat_budget.nc", engine="h5netcdf", mode="w")

    with pytest.raises(validator.ArtifactValidationError, match="scientific tolerance"):
        validator.validate_run(
            candidate,
            profile_name="fixed-500-300-2021",
            expected_commit=commit,
            expected_cache=cache,
            reference_dir=reference,
        )


def test_validate_run_rejects_scientific_difference(tmp_path):
    candidate, cache, commit = _write_run(tmp_path / "candidate", "fixed-500-300-2021")
    reference, _, _ = _write_run(tmp_path / "reference", "fixed-500-300-2021")
    with xr.open_dataset(candidate / "heat_budget.nc", engine="h5netcdf") as opened:
        changed = opened.load()
    changed["adiabatic_term"][0] = 1.0
    changed.to_netcdf(candidate / "heat_budget.nc", engine="h5netcdf", mode="w")

    with pytest.raises(validator.ArtifactValidationError, match="diabatic_residual"):
        validator.validate_run(
            candidate,
            profile_name="fixed-500-300-2021",
            expected_commit=commit,
            expected_cache=cache,
            reference_dir=reference,
        )
