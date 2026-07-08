import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import xarray as xr

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.terms import (
    compute_benchmark_diagnostic_totals,
    compute_benchmark_output_face_fluxes,
)
from src.plot_diagnostics import fig1_benchmark_mass_continuity, fig5_benchmark_comparison


def test_compute_benchmark_diagnostic_totals_matches_figure_5_1_series():
    full_time = np.arange(5)
    output_time = full_time[1:-1]

    benchmark_mass_fluxes = xr.Dataset(
        {
            "benchmark_mass_flux_net": xr.DataArray(
                [10.0, 20.0, 30.0, 40.0, 50.0],
                dims=("time",),
                coords={"time": full_time},
            )
        }
    )
    benchmark_heat_fluxes = xr.Dataset(
        {
            "benchmark_heat_flux_net": xr.DataArray(
                [100.0, 200.0, 300.0, 400.0, 500.0],
                dims=("time",),
                coords={"time": full_time},
            )
        }
    )
    results = xr.Dataset(
        {
            "T_domain_avg": xr.DataArray(
                [2.0, 3.0, 4.0],
                dims=("time",),
                coords={"time": output_time},
            ),
            "dV_dt_true": xr.DataArray(
                [1.0, 2.0, 3.0],
                dims=("time",),
                coords={"time": output_time},
            ),
        }
    )
    advection_terms = xr.Dataset(
        {
            "net_mass_advection": ("time", [11.0, 22.0, 33.0]),
            "mass_flux_contribution_top": ("time", [1.0, 2.0, 3.0]),
            "mass_flux_contribution_bottom": ("time", [0.5, 1.0, 1.5]),
            "advection_term": ("time", [110.0, 220.0, 330.0]),
            "flux_contribution_top": ("time", [10.0, 20.0, 30.0]),
            "flux_contribution_bottom": ("time", [5.0, 10.0, 15.0]),
        },
        coords={"time": output_time},
    )

    out = compute_benchmark_diagnostic_totals(
        benchmark_mass_fluxes,
        benchmark_heat_fluxes,
        results,
        advection_terms,
    )

    assert set(out.data_vars) == {
        "benchmark_mass_flux_net",
        "calculated_mass_flux_net_lateral",
        "benchmark_heat_flux_net",
        "calculated_heat_flux_net_lateral_full",
        "calculated_heat_flux_net_lateral_full_benchmark_mass",
        "calculated_heat_flux_net_lateral",
        "benchmark_heat_flux_net_lateral_prime",
    }
    npt.assert_array_equal(out["time"], output_time)

    mass_lateral = np.array([9.5, 19.0, 28.5])
    heat_anomaly = np.array([95.0, 190.0, 285.0])
    benchmark_mass_net = np.array([-20.0, -30.0, -40.0])

    npt.assert_allclose(out["benchmark_mass_flux_net"], benchmark_mass_net)
    npt.assert_allclose(out["calculated_mass_flux_net_lateral"], mass_lateral)
    npt.assert_allclose(out["benchmark_heat_flux_net"], [-200.0, -300.0, -400.0])
    npt.assert_allclose(out["calculated_heat_flux_net_lateral"], heat_anomaly)
    npt.assert_allclose(
        out["calculated_heat_flux_net_lateral_full"],
        heat_anomaly + np.array([2.0, 3.0, 4.0]) * mass_lateral,
    )
    npt.assert_allclose(
        out["calculated_heat_flux_net_lateral_full_benchmark_mass"],
        heat_anomaly + np.array([2.0, 3.0, 4.0]) * benchmark_mass_net,
    )
    npt.assert_allclose(
        out["benchmark_heat_flux_net_lateral_prime"],
        np.array([-200.0, -300.0, -400.0])
        - benchmark_mass_net * np.array([2.0, 3.0, 4.0]),
    )
    assert out["benchmark_mass_flux_net"].attrs["diagnostic_figure"] == "5.1"
    assert out["benchmark_heat_flux_net"].attrs["units"] == "K m2 Pa s-1"
    assert out["benchmark_heat_flux_net_lateral_prime"].attrs["formula"] == (
        "benchmark_heat_flux_net - benchmark_mass_flux_net * T_domain_avg"
    )


def test_compute_benchmark_diagnostic_totals_supports_surface_bottom():
    time = np.arange(2)
    benchmark_mass_fluxes = xr.Dataset(
        {"benchmark_mass_flux_net": ("time", [1.0, 2.0])},
        coords={"time": time},
    )
    benchmark_heat_fluxes = xr.Dataset(
        {"benchmark_heat_flux_net": ("time", [3.0, 4.0])},
        coords={"time": time},
    )
    results = xr.Dataset(
        {
            "T_domain_avg": ("time", [10.0, 20.0]),
            "dV_dt_true": ("time", [0.5, 1.5]),
        },
        coords={"time": time},
    )
    advection_terms = xr.Dataset(
        {
            "net_mass_advection": ("time", [5.0, 6.0]),
            "mass_flux_contribution_top": ("time", [1.0, 1.0]),
            "advection_term": ("time", [50.0, 60.0]),
            "flux_contribution_top": ("time", [10.0, 10.0]),
        },
        coords={"time": time},
    )

    out = compute_benchmark_diagnostic_totals(
        benchmark_mass_fluxes,
        benchmark_heat_fluxes,
        results,
        advection_terms,
    )

    npt.assert_allclose(out["calculated_mass_flux_net_lateral"], [4.0, 5.0])
    npt.assert_allclose(out["calculated_heat_flux_net_lateral"], [40.0, 50.0])
    npt.assert_allclose(
        out["benchmark_heat_flux_net_lateral_prime"],
        [-3.0 - (-1.0) * 10.0, -4.0 - (-2.0) * 20.0],
    )


def test_compute_benchmark_output_face_fluxes_matches_output_schema():
    full_time = np.arange(5)
    output_time = xr.DataArray(full_time[1:-1], dims=("time",), name="time")
    faces = ("north", "south", "east", "west")

    benchmark_mass_fluxes = xr.Dataset(
        {
            **{
                f"benchmark_mass_flux_{face}": ("time", np.arange(5, dtype=float))
                for face in faces
            },
            "benchmark_mass_flux_net": ("time", np.arange(10, 15, dtype=float)),
        },
        coords={"time": full_time},
    )
    benchmark_heat_fluxes = xr.Dataset(
        {
            **{
                f"benchmark_heat_flux_{face}": ("time", np.arange(20, 25, dtype=float))
                for face in faces
            },
            "benchmark_heat_flux_net": ("time", np.arange(30, 35, dtype=float)),
        },
        coords={"time": full_time},
    )

    out = compute_benchmark_output_face_fluxes(
        benchmark_mass_fluxes,
        benchmark_heat_fluxes,
        output_time,
    )

    assert set(out.data_vars) == {
        *(f"benchmark_mass_flux_{face}" for face in faces),
        *(f"benchmark_heat_flux_{face}" for face in faces),
    }
    npt.assert_array_equal(out["time"], full_time[1:-1])
    npt.assert_allclose(out["benchmark_mass_flux_north"], [-1.0, -2.0, -3.0])
    npt.assert_allclose(out["benchmark_heat_flux_north"], [-21.0, -22.0, -23.0])
    assert out["benchmark_mass_flux_north"].attrs["sign_convention"] == (
        "positive into domain"
    )
    assert out["benchmark_heat_flux_north"].attrs["units"] == "K m2 Pa s-1"


def test_fig1_benchmark_mass_continuity_plots_benchmark_scatter(tmp_path):
    time = np.arange(3)
    dV_dt = xr.DataArray([-1.0, 0.0, 1.0], dims=("time",), coords={"time": time})
    dV_dt_true = xr.DataArray(
        [-1.5, 0.0, 1.5],
        dims=("time",),
        coords={"time": time},
    )
    advection_terms = xr.Dataset(
        {"net_mass_advection": ("time", [-0.9, 0.1, 1.1])},
        coords={"time": time},
    )
    benchmark_mass_flux_net = xr.DataArray(
        [-1.4, -0.1, 1.6],
        dims=("time",),
        coords={"time": time},
    )

    fig1_benchmark_mass_continuity(
        dV_dt,
        advection_terms,
        dV_dt_true,
        benchmark_mass_flux_net,
        str(tmp_path),
    )

    assert (tmp_path / "fig1_benchmark_mass_continuity.png").is_file()
    assert (tmp_path / "fig1.1_benchmark_vs_calculated_mass_flux.png").is_file()
    assert (tmp_path / "fig1.2_benchmark_vs_calculated_dV_dt.png").is_file()


def test_fig5_benchmark_comparison_plots_aligned_diagnostic_totals(tmp_path):
    full_time = np.arange(5)
    output_time = full_time[1:-1]
    faces = ("north", "south", "east", "west")

    benchmark_mass_fluxes = xr.Dataset(
        {
            **{
                f"benchmark_mass_flux_{face}": ("time", np.arange(5, dtype=float))
                for face in faces
            },
            "benchmark_mass_flux_net": ("time", np.arange(5, dtype=float)),
        },
        coords={"time": full_time},
    )
    benchmark_heat_fluxes = xr.Dataset(
        {
            **{
                f"benchmark_heat_flux_{face}": ("time", np.arange(5, dtype=float))
                for face in faces
            },
            "benchmark_heat_flux_net": ("time", np.arange(5, dtype=float)),
        },
        coords={"time": full_time},
    )
    results = xr.Dataset(
        {
            "T_domain_avg": ("time", [280.0, 281.0, 282.0]),
            "dV_dt_true": ("time", [0.0, 0.0, 0.0]),
        },
        coords={"time": output_time},
    )
    advection_terms = xr.Dataset(
        {
            **{
                f"mass_flux_contribution_{face}": ("time", np.ones(3))
                for face in faces
            },
            **{
                f"flux_contribution_{face}": ("time", np.ones(3))
                for face in faces
            },
            "net_mass_advection": ("time", np.ones(3)),
            "mass_flux_contribution_top": ("time", np.zeros(3)),
            "advection_term": ("time", np.ones(3)),
            "flux_contribution_top": ("time", np.zeros(3)),
        },
        coords={"time": output_time},
    )

    fig5_benchmark_comparison(
        benchmark_mass_fluxes,
        benchmark_heat_fluxes,
        results,
        advection_terms,
        str(tmp_path),
    )

    assert (tmp_path / "fig5_benchmark_comparison.png").is_file()
    assert (tmp_path / "fig5.1_net_benchmark_comparison.png").is_file()
    assert (
        tmp_path / "fig5.2_benchmark_vs_calculated_heat_flux_lateral_prime.png"
    ).is_file()
