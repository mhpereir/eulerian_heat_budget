import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import budget, grid
from src.terms import (
    compute_benchmark_diagnostic_totals,
    compute_benchmark_output_face_fluxes,
    compute_full_column_benchmark_terms,
    require_full_column_benchmark_domain,
)
from src.plot_diagnostics import (
    fig1_benchmark_mass_continuity,
    fig5_benchmark_comparison,
    fig6_benchmark_heating_comparison,
)
from src.specs import DomainRequest, DomainSpec, SurfaceBehaviour


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


def test_compute_full_column_benchmark_terms_matches_equations():
    time = np.arange(5)
    output_time = xr.DataArray(time[1:-1], dims=("time",), name="time")
    lat = [45.0, 46.0]
    lon = [-125.0, -124.0]
    area = xr.DataArray(
        [[2.0, 3.0], [4.0, 5.0]],
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )
    vithe_values = (
        np.arange(5, dtype=float)[:, None, None]
        * np.full((1, 2, 2), 10.0)
    )
    benchmark_ds = xr.Dataset(
        {
            "vithe": (("time", "lat", "lon"), vithe_values),
            "viec": (("time", "lat", "lon"), np.full((5, 2, 2), 2.0)),
            "vithed": (("time", "lat", "lon"), np.full((5, 2, 2), 3.0)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    T_domain_avg = xr.DataArray(
        [10.0, 10.0, 10.0],
        dims=("time",),
        coords={"time": output_time},
    )
    dV_dt = xr.DataArray(
        [5.0, 5.0, 5.0],
        dims=("time",),
        coords={"time": output_time},
    )
    benchmark_mass_flux_net = xr.DataArray(
        [20.0, 20.0, 20.0],
        dims=("time",),
        coords={"time": output_time},
    )
    domain_volume = xr.DataArray(
        [100.0, 110.0, 120.0, 130.0, 140.0],
        dims=("time",),
        coords={"time": time},
        name="domain_volume_true",
    )
    benchmark_heat_flux_net = xr.DataArray(
        [-30.0, -31.0, -32.0],
        dims=("time",),
        coords={"time": output_time},
    )

    out = compute_full_column_benchmark_terms(
        benchmark_ds,
        area,
        output_time=output_time,
        domain_volume=domain_volume,
        T_domain_avg=T_domain_avg,
        dV_dt=dV_dt,
        benchmark_mass_flux_net=benchmark_mass_flux_net,
        benchmark_heat_flux_net=benchmark_heat_flux_net,
    )

    assert set(out.data_vars) == {
        "benchmark_vithe",
        "benchmark_viec",
        "benchmark_vithed",
        "benchmark_thermal_content",
        "benchmark_storage_term",
        "benchmark_T_domain_avg",
        "benchmark_mean_temperature_storage_term",
        "benchmark_volume_change_storage_term",
        "benchmark_adiabatic_term",
        "benchmark_heat_flux_divergence",
        "benchmark_heat_flux_divergence_from_walls",
        "benchmark_mass_residual",
        "benchmark_residual_heat",
        "benchmark_diabatic_term_physical",
        "benchmark_diabatic_term",
    }
    conversion = 9.806 / 1005.0
    npt.assert_allclose(out["benchmark_vithe"], [140.0, 280.0, 420.0])
    npt.assert_allclose(out["benchmark_viec"], 28.0)
    npt.assert_allclose(out["benchmark_vithed"], 42.0)
    npt.assert_allclose(out["benchmark_storage_term"], 140.0 * conversion)
    expected_mean = (
        np.array([140.0, 280.0, 420.0]) * conversion
        / np.array([110.0, 120.0, 130.0])
    )
    npt.assert_allclose(out["benchmark_T_domain_avg"], expected_mean)
    full_mean = np.arange(5) * 140.0 * conversion / domain_volume.values
    expected_mean_storage = (
        (full_mean[2:] - full_mean[:-2]) / 2.0
        * domain_volume.values[1:-1]
    )
    npt.assert_allclose(
        out["benchmark_mean_temperature_storage_term"],
        expected_mean_storage,
    )
    npt.assert_allclose(
        out["benchmark_volume_change_storage_term"],
        expected_mean * 10.0,
    )
    npt.assert_allclose(out["benchmark_adiabatic_term"], 28.0 * conversion)
    npt.assert_allclose(
        out["benchmark_heat_flux_divergence"],
        42.0 * conversion,
    )
    npt.assert_allclose(
        out["benchmark_heat_flux_divergence_from_walls"],
        [30.0, 31.0, 32.0],
    )
    expected_physical = (140.0 + 42.0 - 28.0) * conversion
    npt.assert_allclose(out["benchmark_diabatic_term_physical"], expected_physical)
    npt.assert_allclose(out["benchmark_mass_residual"], 15.0)
    npt.assert_allclose(out["benchmark_residual_heat"], 150.0)
    npt.assert_allclose(
        out["benchmark_diabatic_term"],
        expected_physical + 150.0,
    )
    assert out["benchmark_vithe"].attrs["source_param_id"] == 162060
    assert out["benchmark_diabatic_term"].attrs["formula"] == (
        "benchmark_diabatic_term_physical + benchmark_residual_heat"
    )


def test_full_column_benchmark_domain_rejects_partial_domains():
    with pytest.raises(ValueError, match="zg_bottom='surface_pressure'"):
        require_full_column_benchmark_domain(
            DomainSpec(
                lat_min=40.0,
                lat_max=50.0,
                lon_min=-130.0,
                lon_max=-120.0,
                zg_top_pressure=100.0,
                zg_bottom="pressure_level",
                zg_bottom_pressure=100000.0,
            )
        )

    with pytest.raises(ValueError, match="zg_top_pressure = 100 Pa"):
        require_full_column_benchmark_domain(
            DomainSpec(
                lat_min=40.0,
                lat_max=50.0,
                lon_min=-130.0,
                lon_max=-120.0,
                zg_top_pressure=70000.0,
                zg_bottom="surface_pressure",
                zg_bottom_pressure=None,
            )
        )

    with pytest.raises(ValueError, match="zg_top_pressure = 100 Pa"):
        require_full_column_benchmark_domain(
            DomainSpec(
                lat_min=40.0,
                lat_max=50.0,
                lon_min=-130.0,
                lon_max=-120.0,
                zg_top_pressure=50.0,
                zg_bottom="surface_pressure",
                zg_bottom_pressure=None,
            )
        )

    require_full_column_benchmark_domain(
        DomainSpec(
            lat_min=40.0,
            lat_max=50.0,
            lon_min=-130.0,
            lon_max=-120.0,
            zg_top_pressure=100.0,
            zg_bottom="surface_pressure",
            zg_bottom_pressure=None,
        )
    )


def test_calculate_budget_merges_full_column_benchmarks_into_output(tmp_path):
    time = np.arange(
        np.datetime64("1940-06-01T00"),
        np.datetime64("1940-06-01T05"),
        np.timedelta64(1, "h"),
    )
    level = np.array([100000.0, 50000.0, 10000.0, 1000.0, 100.0])
    lat = np.arange(40.0, 45.0)
    lon = np.arange(-125.0, -120.0)
    shape_4d = (time.size, level.size, lat.size, lon.size)
    shape_3d = (time.size, lat.size, lon.size)
    source = xr.Dataset(
        {
            "T": (("time", "level", "lat", "lon"), np.full(shape_4d, 280.0)),
            "u": (("time", "level", "lat", "lon"), np.zeros(shape_4d)),
            "v": (("time", "level", "lat", "lon"), np.zeros(shape_4d)),
            "w": (("time", "level", "lat", "lon"), np.zeros(shape_4d)),
            "sp": (("time", "lat", "lon"), np.full(shape_3d, 100000.0)),
        },
        coords={"time": time, "level": level, "lat": lat, "lon": lon},
    )
    request = DomainRequest(
        bbox=(40.0, 44.0, -125.0, -121.0),
        margin_n=1,
        zg_top_pressure=100.0,
        zg_bottom="surface_pressure",
        zg_bottom_pressure=None,
    )
    surface = SurfaceBehaviour(
        allow_bottom_overflow=False,
        use_surface_variables=False,
        surface_variable_mode="none",
    )
    ds_domain, ds_halo, domain = grid.determine_domain(source, request)
    benchmark = xr.Dataset(
        {
            "Fx_heat": (("time", "lat", "lon"), np.zeros(shape_3d)),
            "Fy_heat": (("time", "lat", "lon"), np.zeros(shape_3d)),
            "Fx_mass": (("time", "lat", "lon"), np.zeros(shape_3d)),
            "Fy_mass": (("time", "lat", "lon"), np.zeros(shape_3d)),
            "vithe": (("time", "lat", "lon"), np.full(shape_3d, 1.0e9)),
            "viec": (("time", "lat", "lon"), np.zeros(shape_3d)),
            "vithed": (("time", "lat", "lon"), np.zeros(shape_3d)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )

    out = budget.calculate_budget(
        ds_domain,
        ds_halo,
        domain,
        surface,
        integral_diagnostics_flag=True,
        plot_dir=str(tmp_path),
        plot_flag=False,
        benchmark_ds=benchmark,
    )

    expected = {
        "benchmark_vithe",
        "benchmark_viec",
        "benchmark_vithed",
        "benchmark_thermal_content",
        "benchmark_storage_term",
        "benchmark_T_domain_avg",
        "benchmark_mean_temperature_storage_term",
        "benchmark_volume_change_storage_term",
        "benchmark_adiabatic_term",
        "benchmark_heat_flux_divergence",
        "benchmark_heat_flux_divergence_from_walls",
        "benchmark_mass_residual",
        "benchmark_residual_heat",
        "benchmark_diabatic_term_physical",
        "benchmark_diabatic_term",
        "calculated_diabatic_term_physical",
        "volume_change_storage_term",
    }
    assert expected.issubset(out.data_vars)
    npt.assert_allclose(out["benchmark_adiabatic_term"], 0.0)
    npt.assert_allclose(out["benchmark_diabatic_term_physical"], 0.0)
    npt.assert_allclose(out["benchmark_diabatic_term"], 0.0)


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
    full_time = np.arange(
        np.datetime64("1940-06-01"),
        np.datetime64("1940-06-06"),
        np.timedelta64(1, "D"),
    )
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


def test_fig6_benchmark_heating_comparison_writes_timeseries_and_scatter(
    tmp_path,
):
    time = np.arange(
        np.datetime64("1940-06-01"),
        np.datetime64("1940-06-04"),
        np.timedelta64(1, "D"),
    )
    results = xr.Dataset(
        {
            "adiabatic_term": ("time", [-2.0, 1.0, 4.0]),
            "diabatic_term": ("time", [3.0, 5.0, 7.0]),
            "calculated_diabatic_term_physical": ("time", [1.0, 2.0, 3.0]),
            "benchmark_adiabatic_term": ("time", [-1.5, 1.5, 3.5]),
            "benchmark_diabatic_term": ("time", [2.5, 5.5, 6.5]),
            "benchmark_diabatic_term_physical": ("time", [1.5, 1.8, 3.2]),
        },
        coords={"time": time},
    )

    fig6_benchmark_heating_comparison(results, str(tmp_path))

    expected = (
        "fig6_benchmark_heating_comparison.png",
        "fig6.1_benchmark_vs_calculated_adiabatic.png",
        "fig6.2_benchmark_vs_calculated_diabatic_workflow.png",
        "fig6.3_benchmark_vs_calculated_diabatic_physical.png",
    )
    for filename in expected:
        assert (tmp_path / filename).is_file()
