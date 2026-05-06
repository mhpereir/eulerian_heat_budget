import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import io
from src.specs import DataSourceConfig, SurfaceBehaviour
from src.time_utils import require_regular_time


def _arco_cfg() -> DataSourceConfig:
    return DataSourceConfig(
        kind="arco_era5",
        arco_path="gs://example-dataset.zarr",
        arco_storage_token="anon",
        time_start="1940-01-01T00:00:00",
        time_end="1940-12-31T23:00:00",
    )


def _arco_dataset(periods: int = 24) -> xr.Dataset:
    time = pd.date_range("1940-06-01T00:00:00", periods=periods, freq="1h")
    values_4d = np.ones((periods, 1, 1, 1))
    values_3d = np.ones((periods, 1, 1))
    return xr.Dataset(
        {
            "temperature": xr.DataArray(
                300.0 * values_4d,
                dims=("time", "pressure_level", "latitude", "longitude"),
            ),
            "u_component_of_wind": xr.DataArray(
                values_4d,
                dims=("time", "pressure_level", "latitude", "longitude"),
            ),
            "v_component_of_wind": xr.DataArray(
                values_4d,
                dims=("time", "pressure_level", "latitude", "longitude"),
            ),
            "vertical_velocity": xr.DataArray(
                0.0 * values_4d,
                dims=("time", "pressure_level", "latitude", "longitude"),
            ),
            "surface_pressure": xr.DataArray(
                100000.0 * values_3d,
                dims=("time", "latitude", "longitude"),
            ),
        },
        coords={
            "time": time,
            "pressure_level": [1000.0],
            "latitude": [45.0],
            "longitude": [230.0],
        },
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


def test_standardize_era5_dataset_applies_temporal_sampling_before_chunking():
    cfg = DataSourceConfig(
        kind="arco_era5",
        arco_path="gs://example-dataset.zarr",
        time_start="1940-06-01T00:00:00",
        time_end="1940-06-01T23:00:00",
        temporal_stride_hours=6,
        temporal_phase_hour=2,
    )
    dataset = _arco_dataset().rename(
        {
            "temperature": "T",
            "u_component_of_wind": "u",
            "v_component_of_wind": "v",
            "vertical_velocity": "w",
            "surface_pressure": "sp",
        }
    )

    out = io.standardize_era5_dataset(dataset, cfg)

    assert out["time"].dt.hour.values.tolist() == [2, 8, 14, 20]
    assert require_regular_time(out["time"]) == 21600.0
    assert out.sizes["time"] == 4


def test_apply_temporal_sampling_requires_stride_and_phase_together():
    cfg = DataSourceConfig(
        kind="arco_era5",
        temporal_stride_hours=6,
        temporal_phase_hour=None,
    )

    with pytest.raises(ValueError, match="must both be set"):
        io.apply_temporal_sampling(xr.Dataset(coords={"time": pd.date_range("1940-01-01", periods=4, freq="1h")}), cfg)


def test_load_arco_benchmark_fluxes_applies_temporal_sampling(monkeypatch):
    dataset = xr.Dataset(
        {
            "vertical_integral_of_eastward_heat_flux": xr.DataArray(
                np.ones((24, 1, 1)),
                dims=("time", "latitude", "longitude"),
            ),
        },
        coords={
            "time": pd.date_range("1940-06-01T00:00:00", periods=24, freq="1h"),
            "latitude": [45.0],
            "longitude": [230.0],
        },
    )
    cfg = DataSourceConfig(
        kind="arco_era5",
        arco_path="gs://example-dataset.zarr",
        arco_storage_token="anon",
        time_start="1940-06-01T00:00:00",
        time_end="1940-06-01T23:00:00",
        temporal_stride_hours=6,
        temporal_phase_hour=4,
    )

    monkeypatch.setattr(io.xr, "open_zarr", lambda *args, **kwargs: dataset)

    out = io.load_arco_benchmark_fluxes(
        cfg,
        {"vertical_integral_of_eastward_heat_flux": "Fx_heat"},
    )

    assert out["time"].dt.hour.values.tolist() == [4, 10, 16, 22]
    assert require_regular_time(out["time"]) == 21600.0


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
