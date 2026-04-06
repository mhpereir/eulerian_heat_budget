import sys
from pathlib import Path

import pytest
import xarray as xr

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import io
from src.specs import DataSourceConfig, SurfaceBehaviour


def _arco_cfg() -> DataSourceConfig:
    return DataSourceConfig(
        kind="arco_era5",
        arco_path="gs://example-dataset.zarr",
        arco_storage_token="anon",
        time_start="1940-01-01T00:00:00",
        time_end="1940-12-31T23:00:00",
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
