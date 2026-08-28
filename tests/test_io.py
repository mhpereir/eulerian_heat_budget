import sys
from pathlib import Path

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


def _request() -> DomainRequest:
    return DomainRequest(
        bbox=(40.0, 45.0, -130.0, -125.0),
        margin_n=1,
        zg_top_pressure=80000.0,
        zg_bottom="pressure_level",
        zg_bottom_pressure=100000.0,
    )


class _FakeSession:
    def __init__(self):
        self.closed = False
        self.close_calls = []


class _FakeFilesystem:
    asynchronous = True
    loop = object()

    def __init__(self):
        self.session = _FakeSession()

    @staticmethod
    def close_session(loop, session, asynchronous=False):
        session.close_calls.append((loop, asynchronous))
        session.closed = True


class _FakeFsspecStore:
    def __init__(self):
        self.fs = _FakeFilesystem()
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def _patch_arco_stores(monkeypatch):
    stores = []
    calls = []

    def fake_from_url(url, *, storage_options, read_only):
        store = _FakeFsspecStore()
        stores.append(store)
        calls.append((url, storage_options, read_only))
        return store

    monkeypatch.setattr(io.FsspecStore, "from_url", fake_from_url)
    return stores, calls


def test_open_arco_zarr_retries_transient_errors(monkeypatch):
    calls = {"count": 0}
    stores, store_calls = _patch_arco_stores(monkeypatch)
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
    assert len(stores) == 3
    assert [store.close_calls for store in stores] == [1, 1, 0]
    assert [store.fs.session.closed for store in stores] == [True, True, False]
    assert [store.fs.session.close_calls for store in stores] == [
        [(stores[0].fs.loop, True)],
        [(stores[1].fs.loop, True)],
        [],
    ]
    assert store_calls == [
        (
            "gs://example-dataset.zarr",
            {"token": "anon", "skip_instance_cache": True},
            True,
        )
    ] * 3

    out.close()

    assert stores[-1].close_calls == 1
    assert stores[-1].fs.session.closed is True
    assert stores[-1].fs.session.close_calls == [(stores[-1].fs.loop, True)]


def test_open_arco_zarr_does_not_retry_non_transient_errors(monkeypatch):
    calls = {"count": 0}
    stores, _ = _patch_arco_stores(monkeypatch)

    def fake_open_zarr(*args, **kwargs):
        calls["count"] += 1
        raise ValueError("bad path")

    monkeypatch.setattr(io.xr, "open_zarr", fake_open_zarr)
    monkeypatch.setattr(io.time, "sleep", lambda seconds: None)

    with pytest.raises(ValueError, match="bad path"):
        io._open_arco_zarr_with_retry(_arco_cfg())

    assert calls["count"] == 1
    assert stores[0].close_calls == 1
    assert stores[0].fs.session.closed is True


def test_open_arco_zarr_accepts_native_chunk_strategy(monkeypatch):
    observed = {}
    stores, _ = _patch_arco_stores(monkeypatch)
    dataset = xr.Dataset(
        {"temperature": xr.DataArray([1.0], dims=("time",))},
        coords={"time": [0]},
    )

    def fake_open_zarr(*args, **kwargs):
        observed.update(kwargs)
        return dataset

    monkeypatch.setattr(io.xr, "open_zarr", fake_open_zarr)

    out = io._open_arco_zarr_with_retry(_arco_cfg(), chunks={})

    assert out is dataset
    assert observed["chunks"] == {}

    out.close()

    assert stores[0].close_calls == 1
    assert stores[0].fs.session.closed is True


def test_standardize_can_preserve_source_chunks(monkeypatch):
    dataset = xr.Dataset(
        {
            "T": xr.DataArray(
                [[[[300.0]]]],
                dims=("time", "level", "lat", "lon"),
                attrs={"units": "K"},
            ),
            "u": xr.DataArray([[[[1.0]]]], dims=("time", "level", "lat", "lon")),
            "v": xr.DataArray([[[[2.0]]]], dims=("time", "level", "lat", "lon")),
            "w": xr.DataArray([[[[0.0]]]], dims=("time", "level", "lat", "lon")),
            "sp": xr.DataArray([[[100000.0]]], dims=("time", "lat", "lon")),
        },
        coords={
            "time": ["1940-06-01T00:00:00"],
            "level": [100000.0],
            "lat": [45.0],
            "lon": [-130.0],
        },
    )
    dataset["level"].attrs["units"] = "Pa"

    def fail_chunk(*args, **kwargs):
        pytest.fail("standardization unexpectedly rechunked the source dataset")

    monkeypatch.setattr(xr.Dataset, "chunk", fail_chunk)

    out = io.standardize_era5_dataset(dataset, _arco_cfg(), rechunk=False)

    assert set(out.data_vars) == {"T", "u", "v", "w", "sp"}


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
        _surface_specs(),
    )

    assert "T" in out
    assert "sp" in out
    assert calls["count"] == 2


def test_load_dataset_staged_cache_requires_request():
    cfg = DataSourceConfig(kind="staged_arco_cache", staged_cache_root="/tmp/ehb-cache")

    with pytest.raises(ValueError, match="DomainRequest"):
        io.load_dataset(cfg, _surface_specs())


def test_load_dataset_staged_cache_uses_local_cache(monkeypatch):
    cfg = DataSourceConfig(
        kind="staged_arco_cache",
        staged_cache_root="/tmp/ehb-cache",
        time_start="1940-06-01T00:00:00",
        time_end="1940-06-01T02:00:00",
    )
    dataset = xr.Dataset(
        {
            "T": xr.DataArray(
                [[[[300.0, 301.0], [302.0, 303.0]]]],
                dims=("time", "level", "lat", "lon"),
                attrs={"units": "K"},
            ),
            "u": xr.DataArray(
                [[[[1.0, 1.0], [1.0, 1.0]]]],
                dims=("time", "level", "lat", "lon"),
            ),
            "v": xr.DataArray(
                [[[[2.0, 2.0], [2.0, 2.0]]]],
                dims=("time", "level", "lat", "lon"),
            ),
            "w": xr.DataArray(
                [[[[0.0, 0.0], [0.0, 0.0]]]],
                dims=("time", "level", "lat", "lon"),
            ),
            "sp": xr.DataArray(
                [[[100000.0, 100000.0], [100000.0, 100000.0]]],
                dims=("time", "lat", "lon"),
            ),
        },
        coords={
            "time": ["1940-06-01T00:00:00"],
            "level": [100000.0],
            "lat": [40.0, 41.0],
            "lon": [-130.0, -129.0],
        },
    )
    cache_calls = []

    monkeypatch.setattr(
        io.arco_cache,
        "load_cache_dataset",
        lambda *args, **kwargs: cache_calls.append((args, kwargs)) or dataset,
    )
    monkeypatch.setattr(
        io,
        "_open_arco_zarr_with_retry",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ARCO should not be opened")),
    )

    out = io.load_dataset(cfg, _surface_specs(), _request())

    assert "T" in out
    assert cache_calls
    assert cache_calls[0][0][0] == "/tmp/ehb-cache"
