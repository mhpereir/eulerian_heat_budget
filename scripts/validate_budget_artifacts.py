#!/usr/bin/env python3
"""Validate EHB run-budget artifacts and optionally compare a reference run."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import xarray as xr


class ArtifactValidationError(RuntimeError):
    """Raised when a run-budget artifact violates its acceptance contract."""


@dataclass(frozen=True)
class ValidationProfile:
    time_start: str
    time_end: str
    time_count: int
    input_start: str
    input_end: str
    zg_top_pressure: float
    zg_bottom: str
    zg_bottom_pressure: float | None
    allow_bottom_overflow: bool
    benchmark_variables: bool
    png_count: int
    required_pngs: tuple[str, ...]


PROFILES = {
    "fixed-500-300-1941": ValidationProfile(
        time_start="1941-05-01T01:00:00",
        time_end="1941-08-31T23:00:00",
        time_count=2951,
        input_start="1941-05-01T00:00:00",
        input_end="1941-09-01T00:00:00",
        zg_top_pressure=30000.0,
        zg_bottom="pressure_level",
        zg_bottom_pressure=50000.0,
        allow_bottom_overflow=False,
        benchmark_variables=False,
        png_count=10,
        required_pngs=(
            "fig2_mass_residual_time_series.png",
            "fig2.2_mass_advection_terms_timeseries.png",
        ),
    ),
    "fixed-500-300-2021": ValidationProfile(
        time_start="2021-05-01T01:00:00",
        time_end="2021-08-31T23:00:00",
        time_count=2951,
        input_start="2021-05-01T00:00:00",
        input_end="2021-09-01T00:00:00",
        zg_top_pressure=30000.0,
        zg_bottom="pressure_level",
        zg_bottom_pressure=50000.0,
        allow_bottom_overflow=False,
        benchmark_variables=False,
        png_count=10,
        required_pngs=(
            "fig2_mass_residual_time_series.png",
            "fig2.2_mass_advection_terms_timeseries.png",
        ),
    ),
    "full-column-benchmark-1941": ValidationProfile(
        time_start="1941-06-01T01:00:00",
        time_end="1941-08-31T22:00:00",
        time_count=2206,
        input_start="1941-06-01T00:00:00",
        input_end="1941-08-31T23:00:00",
        zg_top_pressure=100.0,
        zg_bottom="surface_pressure",
        zg_bottom_pressure=None,
        allow_bottom_overflow=True,
        benchmark_variables=True,
        png_count=19,
        required_pngs=(
            "fig2_mass_residual_time_series.png",
            "fig2.2_mass_advection_terms_timeseries.png",
            "fig5_benchmark_comparison.png",
            "fig6_benchmark_heating_comparison.png",
        ),
    ),
    "full-column-benchmark-2021": ValidationProfile(
        time_start="2021-06-01T01:00:00",
        time_end="2021-08-31T22:00:00",
        time_count=2206,
        input_start="2021-06-01T00:00:00",
        input_end="2021-08-31T23:00:00",
        zg_top_pressure=100.0,
        zg_bottom="surface_pressure",
        zg_bottom_pressure=None,
        allow_bottom_overflow=True,
        benchmark_variables=True,
        png_count=19,
        required_pngs=(
            "fig2_mass_residual_time_series.png",
            "fig2.2_mass_advection_terms_timeseries.png",
            "fig5_benchmark_comparison.png",
            "fig6_benchmark_heating_comparison.png",
        ),
    ),
}

CORE_REQUIRED = (
    "d_dt_T",
    "dT_dt",
    "dT_dt_2",
    "volume_change_storage_term",
    "dV_dt",
    "dV_dt_true",
    "advection_error",
    "adiabatic_term",
    "diabatic_term",
    "residual_heat",
    "T_domain_avg",
    "domain_volume",
    "domain_volume_true",
    "T_scale",
    "advection_term",
    "net_mass_advection",
    "flux_contribution_west",
    "mass_flux_contribution_west",
    "flux_contribution_east",
    "mass_flux_contribution_east",
    "flux_contribution_south",
    "mass_flux_contribution_south",
    "flux_contribution_north",
    "mass_flux_contribution_north",
    "flux_contribution_top",
    "mass_flux_contribution_top",
    "abs_mass_advection_residual_fraction",
)

BENCHMARK_REQUIRED = (
    "benchmark_mass_flux_net",
    "calculated_mass_flux_net_lateral",
    "benchmark_heat_flux_net",
    "calculated_heat_flux_net_lateral_full",
    "calculated_heat_flux_net_lateral_full_benchmark_mass",
    "calculated_heat_flux_net_lateral",
    "benchmark_heat_flux_net_lateral_prime",
    "benchmark_mass_flux_north",
    "benchmark_mass_flux_south",
    "benchmark_mass_flux_east",
    "benchmark_mass_flux_west",
    "benchmark_heat_flux_north",
    "benchmark_heat_flux_south",
    "benchmark_heat_flux_east",
    "benchmark_heat_flux_west",
    "benchmark_vithe",
    "benchmark_viec",
    "benchmark_vithed",
    "benchmark_vimad",
    "benchmark_thermal_content",
    "benchmark_storage_term",
    "benchmark_T_domain_avg",
    "benchmark_mean_temperature_storage_term",
    "benchmark_volume_change_storage_term",
    "benchmark_adiabatic_term",
    "benchmark_heat_flux_divergence",
    "benchmark_heat_flux_divergence_from_walls",
    "benchmark_mass_flux_divergence",
    "benchmark_mass_flux_divergence_from_walls",
    "benchmark_mass_residual",
    "benchmark_residual_heat",
    "benchmark_diabatic_term_physical",
    "benchmark_diabatic_term",
    "calculated_diabatic_term_physical",
)

REFERENCE_RELATIVE_TOLERANCE = 1.0e-11
REFERENCE_ABSOLUTE_TOLERANCE = 1.0e-10
REFERENCE_VOLUME_DERIVATIVE_ULPS = 64.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactValidationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_run_info(run_dir: Path) -> dict:
    path = run_dir / "run_info.json"
    _require(path.is_file() and path.stat().st_size > 0, f"Missing run metadata: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _validate_provenance(
    info: dict,
    profile: ValidationProfile,
    *,
    expected_commit: str,
    expected_cache: Path,
) -> None:
    request = info.get("request", {})
    source = info.get("source_spec", {})
    surface = info.get("surface_behaviour", {})
    git = info.get("git", {})
    cli_args = info.get("cli_args", {})

    _require(request.get("bbox") == [40, 60, -130, -110], "Unexpected PNW bbox.")
    _require(request.get("margin_n") == 1, "Unexpected margin_n.")
    _require(
        request.get("zg_top_pressure") == profile.zg_top_pressure,
        "Unexpected top pressure.",
    )
    _require(request.get("zg_bottom") == profile.zg_bottom, "Unexpected bottom mode.")
    _require(
        request.get("zg_bottom_pressure") == profile.zg_bottom_pressure,
        "Unexpected fixed bottom pressure.",
    )
    _require(source.get("kind") == "staged_arco_cache", "Unexpected data source.")
    _require(
        source.get("time_start") == profile.input_start,
        "Unexpected input start time.",
    )
    _require(
        source.get("time_end") == profile.input_end,
        "Unexpected input end time.",
    )
    _require(
        Path(source.get("staged_cache_root", "")) == expected_cache,
        "Unexpected staged-cache root.",
    )
    _require(
        surface.get("allow_bottom_overflow") is profile.allow_bottom_overflow,
        "Unexpected bottom-overflow setting.",
    )
    _require(
        surface.get("use_surface_variables") is False,
        "Surface variables must be disabled.",
    )
    _require(
        surface.get("surface_variable_mode") == "none",
        "Unexpected surface-variable mode.",
    )
    _require(git.get("commit") == expected_commit, "Unexpected Git commit.")
    _require(git.get("dirty") is False, "Runtime Git checkout was dirty.")
    _require(cli_args.get("region") == "pnw_bartusek", "Unexpected region.")
    _require(
        cli_args.get("include_benchmark_variables") is profile.benchmark_variables,
        "Unexpected benchmark-variable setting.",
    )
    _require(
        cli_args.get("diagnostic_plots") is True,
        "Diagnostic plots were disabled.",
    )
    _require(
        cli_args.get("constant_temperature_test") is False,
        "Constant-temperature test was enabled.",
    )
    _require(cli_args.get("write_netcdf") is True, "NetCDF output was disabled.")


def _validate_identity(
    name: str,
    actual: xr.DataArray,
    expected: xr.DataArray,
) -> dict[str, float]:
    difference = np.asarray((actual - expected).values, dtype=np.float64)
    maximum = float(np.max(np.abs(difference)))
    scale = float(
        max(
            np.max(np.abs(np.asarray(actual.values, dtype=np.float64))),
            np.max(np.abs(np.asarray(expected.values, dtype=np.float64))),
            1.0,
        )
    )
    tolerance = float(max(1.0e-10, 8.0 * np.spacing(scale)))
    _require(
        maximum <= tolerance,
        f"Identity {name!r} residual {maximum:g} exceeds {tolerance:g}.",
    )
    return {"max_abs_residual": maximum, "tolerance": tolerance}


def _sum_present(ds: xr.Dataset, prefix: str) -> xr.DataArray:
    names = [
        f"{prefix}_{face}"
        for face in ("west", "east", "south", "north", "top", "bottom")
        if f"{prefix}_{face}" in ds
    ]
    _require(len(names) >= 5, f"Incomplete face-term inventory for {prefix!r}.")
    return sum((ds[name] for name in names[1:]), start=ds[names[0]])


def _validate_dataset(ds: xr.Dataset, profile: ValidationProfile) -> dict:
    required = list(CORE_REQUIRED)
    if not profile.benchmark_variables:
        required.extend(("flux_contribution_bottom", "mass_flux_contribution_bottom"))
    else:
        required.extend(BENCHMARK_REQUIRED)
    missing = sorted(set(required) - set(ds.data_vars))
    _require(not missing, f"Missing required variables: {missing}")

    _require(ds.sizes.get("time") == profile.time_count, "Unexpected time-sample count.")
    time = np.asarray(ds["time"].values).astype("datetime64[ns]")
    _require(time[0] == np.datetime64(profile.time_start), "Unexpected first output time.")
    _require(time[-1] == np.datetime64(profile.time_end), "Unexpected last output time.")
    _require(
        np.all(np.diff(time) == np.timedelta64(1, "h")),
        "Output time is not regular hourly data.",
    )

    nonfinite = [
        name
        for name in ds.data_vars
        if not np.isfinite(np.asarray(ds[name].values)).all()
    ]
    _require(not nonfinite, f"Nonfinite output variables: {nonfinite}")
    _require(bool((ds["domain_volume"] > 0).all()), "domain_volume must be positive.")
    _require(bool((ds["domain_volume_true"] > 0).all()), "domain_volume_true must be positive.")
    _require(
        100.0 < float(ds["T_domain_avg"].min()) < 350.0,
        "Domain-average temperature minimum is implausible.",
    )
    _require(
        100.0 < float(ds["T_domain_avg"].max()) < 350.0,
        "Domain-average temperature maximum is implausible.",
    )
    _require(0.0 < float(ds["T_scale"]) < 200.0, "T_scale is implausible.")
    _require(
        bool((ds["abs_mass_advection_residual_fraction"] >= 0).all()),
        "Absolute residual fraction is negative.",
    )

    identities = {
        "advection_from_faces": _validate_identity(
            "advection_from_faces",
            ds["advection_term"],
            _sum_present(ds, "flux_contribution"),
        ),
        "mass_advection_from_faces": _validate_identity(
            "mass_advection_from_faces",
            ds["net_mass_advection"],
            _sum_present(ds, "mass_flux_contribution"),
        ),
        "diabatic_residual": _validate_identity(
            "diabatic_residual",
            ds["diabatic_term"],
            ds["dT_dt"] - ds["advection_term"] - ds["adiabatic_term"],
        ),
        "residual_heat": _validate_identity(
            "residual_heat",
            ds["residual_heat"],
            (ds["net_mass_advection"] - ds["dV_dt"]) * ds["T_domain_avg"],
        ),
        "advection_error": _validate_identity(
            "advection_error",
            ds["advection_error"],
            (ds["net_mass_advection"] - ds["dV_dt"]) * ds["T_scale"],
        ),
        "volume_change_storage": _validate_identity(
            "volume_change_storage",
            ds["volume_change_storage_term"],
            ds["T_domain_avg"] * ds["dV_dt"],
        ),
        "mean_temperature_storage": _validate_identity(
            "mean_temperature_storage",
            ds["dT_dt_2"],
            ds["d_dt_T"] - ds["T_domain_avg"] * ds["dV_dt"],
        ),
    }

    if profile.benchmark_variables:
        benchmark_identities: dict[str, tuple[xr.DataArray, xr.DataArray]] = {
            "calculated_diabatic_physical": (
                ds["calculated_diabatic_term_physical"],
                ds["diabatic_term"] - ds["residual_heat"],
            ),
            "benchmark_heat_divergence_from_walls": (
                ds["benchmark_heat_flux_divergence_from_walls"],
                -ds["benchmark_heat_flux_net"],
            ),
            "benchmark_mass_divergence_from_walls": (
                ds["benchmark_mass_flux_divergence_from_walls"],
                -ds["benchmark_mass_flux_net"],
            ),
            "benchmark_mass_residual": (
                ds["benchmark_mass_residual"],
                ds["benchmark_mass_flux_net"] - ds["dV_dt_true"],
            ),
            "benchmark_residual_heat": (
                ds["benchmark_residual_heat"],
                ds["benchmark_mass_residual"] * ds["T_domain_avg"],
            ),
            "benchmark_diabatic_physical": (
                ds["benchmark_diabatic_term_physical"],
                ds["benchmark_storage_term"]
                + ds["benchmark_heat_flux_divergence"]
                - ds["benchmark_adiabatic_term"],
            ),
            "benchmark_diabatic_workflow": (
                ds["benchmark_diabatic_term"],
                ds["benchmark_diabatic_term_physical"]
                + ds["benchmark_residual_heat"],
            ),
        }
        for identity_name, (actual, expected) in benchmark_identities.items():
            identities[identity_name] = _validate_identity(
                identity_name,
                actual,
                expected,
            )

    return {
        "time_count": profile.time_count,
        "time_start": str(time[0]),
        "time_end": str(time[-1]),
        "variable_count": len(ds.data_vars),
        "identities": identities,
        "T_domain_avg_min": float(ds["T_domain_avg"].min()),
        "T_domain_avg_max": float(ds["T_domain_avg"].max()),
        "T_scale": float(ds["T_scale"]),
    }


def _png_inventory(run_dir: Path, profile: ValidationProfile) -> list[Path]:
    plot_dir = run_dir / "plots"
    pngs = sorted(plot_dir.rglob("*.png")) if plot_dir.is_dir() else []
    _require(
        len(pngs) == profile.png_count,
        f"Expected {profile.png_count} PNGs, found {len(pngs)}.",
    )
    names = {path.name for path in pngs}
    missing = sorted(set(profile.required_pngs) - names)
    _require(not missing, f"Missing required plots: {missing}")
    _require(all(path.stat().st_size > 0 for path in pngs), "One or more PNGs are empty.")
    return pngs


def _compare_reference(
    candidate: xr.Dataset,
    candidate_pngs: list[Path],
    reference_dir: Path,
    profile: ValidationProfile,
) -> dict:
    reference_path = reference_dir / "heat_budget.nc"
    _require(reference_path.is_file(), f"Missing reference NetCDF: {reference_path}")
    with xr.open_dataset(reference_path, engine="h5netcdf") as opened:
        reference = opened.load()
    _require(candidate.sizes == reference.sizes, "Candidate dimensions differ from reference.")
    _require(
        set(candidate.coords) == set(reference.coords),
        "Candidate coordinates differ from reference.",
    )
    for name in candidate.coords:
        _require(
            candidate[name].identical(reference[name]),
            f"Candidate coordinate {name!r} differs from reference.",
        )
    _require(
        set(candidate.data_vars) == set(reference.data_vars),
        "Candidate variable inventory differs from reference.",
    )
    _require(
        candidate.attrs == reference.attrs,
        "Candidate global attributes differ from reference.",
    )

    time = np.asarray(reference["time"].values).astype("datetime64[ns]")
    timestep_seconds = float(
        np.median(np.diff(time).astype("timedelta64[ns]").astype(np.float64)) / 1.0e9
    )
    volume_scale = float(
        max(
            np.max(np.abs(np.asarray(candidate["domain_volume"], dtype=np.float64))),
            np.max(np.abs(np.asarray(reference["domain_volume"], dtype=np.float64))),
        )
    )
    volume_derivative_tolerance = float(
        REFERENCE_VOLUME_DERIVATIVE_ULPS
        * np.spacing(volume_scale)
        / timestep_seconds
    )
    temperature_scale = float(
        max(
            np.max(np.abs(np.asarray(candidate["T_domain_avg"], dtype=np.float64))),
            np.max(np.abs(np.asarray(reference["T_domain_avg"], dtype=np.float64))),
        )
    )
    absolute_tolerances = {
        "dV_dt": volume_derivative_tolerance,
        "volume_change_storage_term": volume_derivative_tolerance * temperature_scale,
    }

    comparisons = {}
    dataset_identical = True
    for name in sorted(candidate.data_vars):
        candidate_var = candidate[name]
        reference_var = reference[name]
        _require(
            candidate_var.dims == reference_var.dims,
            f"Candidate variable {name!r} dimensions differ from reference.",
        )
        _require(
            candidate_var.dtype == reference_var.dtype,
            f"Candidate variable {name!r} dtype differs from reference.",
        )
        _require(
            candidate_var.attrs == reference_var.attrs,
            f"Candidate variable {name!r} attributes differ from reference.",
        )
        candidate_values = np.asarray(candidate_var.values)
        reference_values = np.asarray(reference_var.values)
        identical = bool(np.array_equal(candidate_values, reference_values, equal_nan=True))
        dataset_identical = dataset_identical and identical
        difference = np.abs(
            candidate_values.astype(np.float64) - reference_values.astype(np.float64)
        )
        maximum_difference = float(np.max(difference))
        scale = float(
            max(
                np.max(np.abs(candidate_values.astype(np.float64))),
                np.max(np.abs(reference_values.astype(np.float64))),
                1.0,
            )
        )
        absolute_tolerance = float(
            absolute_tolerances.get(name, REFERENCE_ABSOLUTE_TOLERANCE)
        )
        tolerance = float(
            max(absolute_tolerance, REFERENCE_RELATIVE_TOLERANCE * scale)
        )
        _require(
            maximum_difference <= tolerance,
            f"Candidate variable {name!r} differs from reference by "
            f"{maximum_difference:g}, which exceeds scientific tolerance {tolerance:g}.",
        )
        comparisons[name] = {
            "identical": identical,
            "max_abs_difference": maximum_difference,
            "scale": scale,
            "tolerance": tolerance,
        }

    candidate_by_name = {path.name: path for path in candidate_pngs}
    reference_by_name = {path.name: path for path in (reference_dir / "plots").rglob("*.png")}
    compared = []
    for name in profile.required_pngs:
        _require(name in reference_by_name, f"Reference plot is missing: {name}")
        with (
            Image.open(candidate_by_name[name]) as candidate_image,
            Image.open(reference_by_name[name]) as reference_image,
        ):
            _require(
                np.array_equal(np.asarray(candidate_image), np.asarray(reference_image)),
                f"Candidate plot pixels differ from reference for {name}.",
            )
        compared.append(name)
    return {
        "dataset_identical": dataset_identical,
        "dataset_scientifically_equivalent": True,
        "relative_tolerance": REFERENCE_RELATIVE_TOLERANCE,
        "variables": comparisons,
        "pixel_identical_plots": compared,
        "reference_netcdf_sha256": _sha256(reference_path),
    }


def validate_run(
    run_dir: Path,
    *,
    profile_name: str,
    expected_commit: str,
    expected_cache: Path,
    reference_dir: Path | None = None,
) -> dict:
    profile = PROFILES[profile_name]
    run_dir = run_dir.resolve()
    netcdf_path = run_dir / "heat_budget.nc"
    _require(
        netcdf_path.is_file() and netcdf_path.stat().st_size > 0,
        f"Missing NetCDF: {netcdf_path}",
    )
    info = _load_run_info(run_dir)
    _validate_provenance(
        info,
        profile,
        expected_commit=expected_commit,
        expected_cache=expected_cache,
    )
    with xr.open_dataset(netcdf_path, engine="h5netcdf") as opened:
        dataset = opened.load()
    scientific = _validate_dataset(dataset, profile)
    pngs = _png_inventory(run_dir, profile)

    report = {
        "passed": True,
        "profile": profile_name,
        "run_id": info.get("run_id"),
        "commit": expected_commit,
        "run_dir": str(run_dir),
        "cache": str(expected_cache),
        "netcdf_sha256": _sha256(netcdf_path),
        "scientific": scientific,
        "png_count": len(pngs),
        "png_sha256": {path.name: _sha256(path) for path in pngs},
    }
    if reference_dir is not None:
        report["reference_comparison"] = _compare_reference(
            dataset,
            pngs,
            reference_dir.resolve(),
            profile,
        )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-cache", type=Path, required=True)
    parser.add_argument("--reference-run-dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    _require(not args.report.exists(), f"Refusing to overwrite report: {args.report}")
    try:
        report = validate_run(
            args.run_dir,
            profile_name=args.profile,
            expected_commit=args.expected_commit,
            expected_cache=args.expected_cache,
            reference_dir=args.reference_run_dir,
        )
    except Exception as exc:
        report = {
            "passed": False,
            "profile": args.profile,
            "run_dir": str(args.run_dir),
            "error": f"{type(exc).__name__}: {exc}",
        }
        status = 1
    else:
        status = 0

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
