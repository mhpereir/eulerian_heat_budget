import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import io
from src.specs import DataSourceConfig, DomainRequest, SurfaceBehaviour


def _arco_cfg() -> DataSourceConfig:
    return DataSourceConfig(
        kind="arco_era5",
        arco_path="gs://example-dataset.zarr",
        arco_storage_token="anon",
        time_start="1940-01-01T00:00:00",
        time_end="1940-12-31T23:00:00",
    )


def _surface_specs() -> SurfaceBehaviour:
    return SurfaceBehaviour(
        allow_bottom_overflow=False,
        use_surface_variables=False,
        surface_variable_mode="none",
    )


def test_open_arco_zarr_retries_transient_errors(monkeypatch):
    calls = {"count": 0}
    dataset = xr.Dataset(
        {
            "temperature": xr.DataArray([1.0], dims=("time",)),
        },
        coords={"time": [0]},
    )

    def fake_open_zarr(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise OSError("Temporary failure in name resolution")
        return dataset

    monkeypatch.setattr(io.xr, "open_zarr", fake_open_zarr)
    monkeypatch.setattr(io.time, "sleep", lambda seconds: None)

    out = io._open_arco_zarr_with_retry(_arco_cfg())

    assert out is dataset
    assert calls["count"] == 3


def test_open_arco_zarr_does_not_retry_non_transient_errors(monkeypatch):
    calls = {"count": 0}

    def fake_open_zarr(*args, **kwargs):
        calls["count"] += 1
        raise ValueError("bad path")

    monkeypatch.setattr(io.xr, "open_zarr", fake_open_zarr)
    monkeypatch.setattr(io.time, "sleep", lambda seconds: None)

    with pytest.raises(ValueError, match="bad path"):
        io._open_arco_zarr_with_retry(_arco_cfg())

    assert calls["count"] == 1


def test_load_arco_benchmark_fluxes_uses_retrying_open(monkeypatch):
    calls = {"count": 0}
    dataset = xr.Dataset(
        {
            "vertical_integral_of_eastward_heat_flux": xr.DataArray(
                [[[1.0]]],
                dims=("time", "latitude", "longitude"),
            ),
        },
        coords={
            "time": ["1940-06-01T00:00:00"],
            "latitude": [45.0],
            "longitude": [230.0],
        },
    )

    def fake_open_zarr(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("Cannot connect to host storage.googleapis.com:443")
        return dataset

    monkeypatch.setattr(io.xr, "open_zarr", fake_open_zarr)
    monkeypatch.setattr(io.time, "sleep", lambda seconds: None)

    out = io.load_arco_benchmark_fluxes(
        _arco_cfg(),
        {"vertical_integral_of_eastward_heat_flux": "Fx_heat"},
    )

    assert "Fx_heat" in out
    assert calls["count"] == 2


def test_load_dataset_retries_arco_open(monkeypatch):
    calls = {"count": 0}
    dataset = xr.Dataset(
        {
            "temperature": xr.DataArray(
                [[[[300.0]]]],
                dims=("time", "pressure_level", "latitude", "longitude"),
            ),
            "u_component_of_wind": xr.DataArray(
                [[[[1.0]]]],
                dims=("time", "pressure_level", "latitude", "longitude"),
            ),
            "v_component_of_wind": xr.DataArray(
                [[[[1.0]]]],
                dims=("time", "pressure_level", "latitude", "longitude"),
            ),
            "vertical_velocity": xr.DataArray(
                [[[[0.0]]]],
                dims=("time", "pressure_level", "latitude", "longitude"),
            ),
            "surface_pressure": xr.DataArray(
                [[[100000.0]]],
                dims=("time", "latitude", "longitude"),
            ),
        },
        coords={
            "time": ["1940-06-01T00:00:00"],
            "pressure_level": [1000.0],
            "latitude": [45.0],
            "longitude": [230.0],
        },
    )

    def fake_open_zarr(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("Temporary failure in name resolution")
        return dataset

    monkeypatch.setattr(io.xr, "open_zarr", fake_open_zarr)
    monkeypatch.setattr(io.time, "sleep", lambda seconds: None)

    out = io.load_dataset(
        _arco_cfg(),
        SurfaceBehaviour(
            allow_bottom_overflow=False,
            use_surface_variables=False,
            surface_variable_mode="none",
        ),
    )

    assert "T" in out
    assert "sp" in out
    assert calls["count"] == 2


def test_load_dataset_opens_staged_zarr_without_arco(monkeypatch, tmp_path):
    staged_path = tmp_path / "subset.zarr"
    staged_path.mkdir()
    times = np.array(
        ["1940-06-01T00:00:00", "1940-06-01T01:00:00", "1940-06-01T02:00:00"],
        dtype="datetime64[ns]",
    )
    levels = np.array([100000.0, 90000.0], dtype=float)
    lat = np.array([40.0, 41.0], dtype=float)
    lon = np.array([-130.0, -129.0], dtype=float)
    shape_4d = (times.size, levels.size, lat.size, lon.size)
    shape_3d = (times.size, lat.size, lon.size)
    ds = xr.Dataset(
        {
            "T": xr.DataArray(np.full(shape_4d, 300.0), dims=("time", "level", "lat", "lon")),
            "u": xr.DataArray(np.full(shape_4d, 1.0), dims=("time", "level", "lat", "lon")),
            "v": xr.DataArray(np.full(shape_4d, 2.0), dims=("time", "level", "lat", "lon")),
            "w": xr.DataArray(np.full(shape_4d, 0.0), dims=("time", "level", "lat", "lon")),
            "sp": xr.DataArray(np.full(shape_3d, 100000.0), dims=("time", "lat", "lon")),
            "Fx_heat": xr.DataArray(np.full(shape_3d, 10.0), dims=("time", "lat", "lon")),
            "Fy_heat": xr.DataArray(np.full(shape_3d, 20.0), dims=("time", "lat", "lon")),
            "Fx_mass": xr.DataArray(np.full(shape_3d, 30.0), dims=("time", "lat", "lon")),
            "Fy_mass": xr.DataArray(np.full(shape_3d, 40.0), dims=("time", "lat", "lon")),
        },
        coords={
            "time": times,
            "level": levels,
            "lat": lat,
            "lon": lon,
            "p_start": ("level", np.array([105000.0, 95000.0])),
            "p_end": ("level", np.array([95000.0, 85000.0])),
        },
    )
    ds["T"].attrs["units"] = "K"

    monkeypatch.setattr(
        io,
        "_open_arco_zarr_with_retry",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ARCO should not be opened")),
    )
    monkeypatch.setattr(io.xr, "open_zarr", lambda *args, **kwargs: ds)

    out = io.load_dataset(
        DataSourceConfig(
            kind="staged_zarr",
            staged_data_path=str(staged_path),
            time_start="1940-06-01T01:00:00",
            time_end="1940-06-01T02:00:00",
        ),
        _surface_specs(),
    )
    benchmark = io.extract_staged_benchmark_fluxes(out)

    assert out["T"].dims == ("time", "level", "lat", "lon")
    assert out["sp"].dims == ("time", "lat", "lon")
    assert out.sizes["time"] == 2
    assert set(benchmark.data_vars) == {"Fx_heat", "Fy_heat", "Fx_mass", "Fy_mass"}


def test_build_arco_staged_subset_renames_crops_and_preserves_pressure_bounds(monkeypatch):
    times = np.array(
        ["1940-06-01T00:00:00", "1940-06-01T01:00:00", "1940-06-01T02:00:00"],
        dtype="datetime64[ns]",
    )
    levels_hpa = np.array([1000.0, 900.0, 800.0, 700.0], dtype=float)
    lat = np.array([39.0, 40.0, 41.0, 42.0, 43.0], dtype=float)
    lon = np.array([229.0, 230.0, 231.0, 232.0], dtype=float)
    shape_4d = (times.size, levels_hpa.size, lat.size, lon.size)
    shape_3d = (times.size, lat.size, lon.size)
    arco_ds = xr.Dataset(
        {
            "temperature": xr.DataArray(np.full(shape_4d, 300.0), dims=("time", "pressure_level", "latitude", "longitude")),
            "u_component_of_wind": xr.DataArray(np.full(shape_4d, 1.0), dims=("time", "pressure_level", "latitude", "longitude")),
            "v_component_of_wind": xr.DataArray(np.full(shape_4d, 2.0), dims=("time", "pressure_level", "latitude", "longitude")),
            "vertical_velocity": xr.DataArray(np.full(shape_4d, 0.0), dims=("time", "pressure_level", "latitude", "longitude")),
            "surface_pressure": xr.DataArray(np.full(shape_3d, 100000.0), dims=("time", "latitude", "longitude")),
            "vertical_integral_of_eastward_heat_flux": xr.DataArray(np.full(shape_3d, 10.0), dims=("time", "latitude", "longitude")),
            "vertical_integral_of_northward_heat_flux": xr.DataArray(np.full(shape_3d, 20.0), dims=("time", "latitude", "longitude")),
            "vertical_integral_of_eastward_mass_flux": xr.DataArray(np.full(shape_3d, 30.0), dims=("time", "latitude", "longitude")),
            "vertical_integral_of_northward_mass_flux": xr.DataArray(np.full(shape_3d, 40.0), dims=("time", "latitude", "longitude")),
        },
        coords={
            "time": times,
            "pressure_level": ("pressure_level", levels_hpa, {"units": "hPa"}),
            "latitude": lat,
            "longitude": lon,
        },
    )
    monkeypatch.setattr(io, "_open_arco_zarr_with_retry", lambda cfg: arco_ds)

    out = io.build_arco_staged_subset(
        _arco_cfg(),
        _surface_specs(),
        DomainRequest(
            bbox=(40.0, 42.0, -130.0, -128.0),
            margin_n=1,
            zg_top_pressure=80000.0,
            zg_bottom="pressure_level",
            zg_bottom_pressure=90000.0,
        ),
        include_benchmark_variables=True,
    )

    assert {"T", "u", "v", "w", "sp", "Fx_heat", "Fy_heat", "Fx_mass", "Fy_mass"}.issubset(out.data_vars)
    np.testing.assert_allclose(out["level"].values, np.array([90000.0, 80000.0]))
    np.testing.assert_allclose(out["p_start"].values, np.array([95000.0, 85000.0]))
    np.testing.assert_allclose(out["p_end"].values, np.array([85000.0, 75000.0]))
    assert out["lon"].min() >= -131.0
    assert out["lon"].max() <= -128.0
