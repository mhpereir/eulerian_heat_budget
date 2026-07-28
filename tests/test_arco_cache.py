import importlib
import multiprocessing
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import budget, cli, grid
from src.specs import DataSourceConfig, DomainRequest, SurfaceBehaviour
from src_arco import cache, selection, variables


staged_arco_retrieval = importlib.import_module("scripts.staged_arco_retrieval")


def _tile_write_metadata() -> cache._TileWriteMetadata:
    return cache._TileWriteMetadata(
        sizes=(("time", 1), ("level", 1), ("lat", 1), ("lon", 1)),
        variables=("dummy",),
        time_start="1940-06-01T00:00:00.000000000",
        time_end="1940-06-01T00:00:00.000000000",
        lat_min=1.0,
        lat_max=1.0,
        lon_min=11.0,
        lon_max=11.0,
        level_min=80000.0,
        level_max=80000.0,
    )


def _spawn_success_worker(send_conn, cache_root, tmp_path, source_cfg, request, include_benchmark_variables):
    (Path(tmp_path) / "child.pid").write_text(str(os.getpid()))
    send_conn.send({"status": "success", "metadata": _tile_write_metadata()})
    send_conn.close()


def _spawn_lingering_success_worker(
    send_conn,
    cache_root,
    tmp_path,
    source_cfg,
    request,
    include_benchmark_variables,
):
    _spawn_success_worker(
        send_conn,
        cache_root,
        tmp_path,
        source_cfg,
        request,
        include_benchmark_variables,
    )
    time.sleep(60)


def _spawn_timeout_worker(send_conn, cache_root, tmp_path, source_cfg, request, include_benchmark_variables):
    (Path(cache_root) / "timeout-child.pid").write_text(str(os.getpid()))
    time.sleep(60)


def _spawn_delayed_success_worker(
    send_conn,
    cache_root,
    tmp_path,
    source_cfg,
    request,
    include_benchmark_variables,
):
    time.sleep(0.2)
    _spawn_success_worker(
        send_conn,
        cache_root,
        tmp_path,
        source_cfg,
        request,
        include_benchmark_variables,
    )


def _spawn_unexpected_exit_worker(
    send_conn,
    cache_root,
    tmp_path,
    source_cfg,
    request,
    include_benchmark_variables,
):
    os._exit(7)


def _request() -> DomainRequest:
    return DomainRequest(
        bbox=(1.0, 5.0, 11.0, 15.0),
        margin_n=1,
        zg_top_pressure=80000.0,
        zg_bottom="pressure_level",
        zg_bottom_pressure=100000.0,
    )


def _source_cfg() -> DataSourceConfig:
    return DataSourceConfig(
        kind="arco_era5",
        arco_path="gs://example.zarr",
        time_start="1940-06-01T00:00:00",
        time_end="1940-06-01T04:00:00",
    )


def _source_cfg_with(
    *,
    time_start: str = "1940-06-01T00:00:00",
    time_end: str = "1940-06-01T04:00:00",
    arco_path: str = "gs://example.zarr",
) -> DataSourceConfig:
    return DataSourceConfig(
        kind="arco_era5",
        arco_path=arco_path,
        time_start=time_start,
        time_end=time_end,
    )


def _staged_source_cfg(
    *,
    time_start: str = "1940-06-01T00:00:00",
    time_end: str = "1940-06-01T04:00:00",
    staged_cache_root: str = "/tmp/ehb-cache",
    arco_path: str = "gs://example.zarr",
) -> DataSourceConfig:
    return DataSourceConfig(
        kind="staged_arco_cache",
        staged_cache_root=staged_cache_root,
        arco_path=arco_path,
        time_start=time_start,
        time_end=time_end,
    )


def _request_with(
    *,
    zg_top_pressure: float = 80000.0,
    zg_bottom: str = "pressure_level",
    zg_bottom_pressure: float | None = 100000.0,
) -> DomainRequest:
    return DomainRequest(
        bbox=(1.0, 5.0, 11.0, 15.0),
        margin_n=1,
        zg_top_pressure=zg_top_pressure,
        zg_bottom=zg_bottom,  # type: ignore[arg-type]
        zg_bottom_pressure=zg_bottom_pressure,
    )


def _patch_to_zarr_creates_store(monkeypatch) -> None:
    def fake_to_zarr(self, path, mode="w"):
        Path(path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(xr.Dataset, "to_zarr", fake_to_zarr)


def _patch_open_zarr_from_registry(monkeypatch, registry: dict[str, xr.Dataset], opened: list[str] | None = None) -> None:
    def fake_open_zarr(path, *args, **kwargs):
        key = str(path)
        if opened is not None:
            opened.append(key)
        return registry[key]

    monkeypatch.setattr(cache.xr, "open_zarr", fake_open_zarr)


def _write_indexed_tile(
    tmp_path,
    source_cfg: DataSourceConfig,
    request: DomainRequest,
    *,
    include_benchmark_variables: bool = False,
    ds: xr.Dataset | None = None,
) -> tuple[Path, xr.Dataset]:
    source_ds = _canonical_dataset() if ds is None else ds
    if source_cfg.time_start is not None or source_cfg.time_end is not None:
        source_ds = source_ds.sel(time=slice(source_cfg.time_start, source_cfg.time_end))
    tile = cache.build_arco_cache_tile(
        source_ds,
        request,
        include_benchmark_variables=include_benchmark_variables,
    )
    tile_path = cache.write_tile(
        tmp_path,
        tile,
        source_cfg,
        request,
        include_benchmark_variables=include_benchmark_variables,
    )
    return tile_path, tile


def _surface_specs() -> SurfaceBehaviour:
    return SurfaceBehaviour(
        allow_bottom_overflow=False,
        use_surface_variables=False,
        surface_variable_mode="none",
    )


def _canonical_dataset() -> xr.Dataset:
    time = np.array(
        [
            "1940-06-01T00:00:00",
            "1940-06-01T01:00:00",
            "1940-06-01T02:00:00",
            "1940-06-01T03:00:00",
            "1940-06-01T04:00:00",
        ],
        dtype="datetime64[ns]",
    )
    level = np.array([100000.0, 90000.0, 80000.0, 70000.0])
    lat = np.arange(0.0, 7.0)
    lon = np.arange(10.0, 17.0)
    shape_4d = (time.size, level.size, lat.size, lon.size)
    shape_3d = (time.size, lat.size, lon.size)
    values = np.arange(np.prod(shape_4d), dtype=float).reshape(shape_4d)

    ds = xr.Dataset(
        {
            "T": xr.DataArray(280.0 + values, dims=("time", "level", "lat", "lon"), attrs={"units": "K"}),
            "u": xr.DataArray(1.0 + values, dims=("time", "level", "lat", "lon")),
            "v": xr.DataArray(2.0 + values, dims=("time", "level", "lat", "lon")),
            "w": xr.DataArray(0.01 + values, dims=("time", "level", "lat", "lon")),
            "sp": xr.DataArray(np.full(shape_3d, 101000.0), dims=("time", "lat", "lon")),
            "Fx_heat": xr.DataArray(np.full(shape_3d, 10.0), dims=("time", "lat", "lon")),
            "Fy_heat": xr.DataArray(np.full(shape_3d, 20.0), dims=("time", "lat", "lon")),
            "Fx_mass": xr.DataArray(np.full(shape_3d, 30.0), dims=("time", "lat", "lon")),
            "Fy_mass": xr.DataArray(np.full(shape_3d, 40.0), dims=("time", "lat", "lon")),
            "vithe": xr.DataArray(np.full(shape_3d, 50.0), dims=("time", "lat", "lon")),
            "viec": xr.DataArray(np.full(shape_3d, 60.0), dims=("time", "lat", "lon")),
            "vithed": xr.DataArray(np.full(shape_3d, 70.0), dims=("time", "lat", "lon")),
        },
        coords={"time": time, "level": level, "lat": lat, "lon": lon},
    )
    ds["level"].attrs["units"] = "Pa"
    return ds


def test_build_arco_cache_tile_stores_wall_only_velocities():
    tile = cache.build_arco_cache_tile(
        _canonical_dataset(),
        _request(),
        include_benchmark_variables=True,
    )

    assert "u" not in tile
    assert "v" not in tile
    assert tile["u_wall"].dims == ("time", "level", "u_lat", "u_lon")
    assert tile["v_wall"].dims == ("time", "level", "v_lat", "v_lon")
    assert set(tile["u_wall"]["u_lat"].values) == {2.0, 3.0, 4.0}
    assert set(tile["u_wall"]["u_lon"].values) == {11.0, 12.0, 14.0, 15.0}
    assert set(tile["v_wall"]["v_lat"].values) == {1.0, 2.0, 4.0, 5.0}
    assert set(tile["v_wall"]["v_lon"].values) == {12.0, 13.0, 14.0}
    assert tile["Fx_heat"].dims == ("time", "u_lat", "u_lon")
    assert tile["Fy_heat"].dims == ("time", "v_lat", "v_lon")
    for name in variables.COLUMN_BENCHMARK_VAR_NAMES:
        assert tile[name].dims == ("time", "lat", "lon")
    assert "p_start" in tile.coords
    assert "Fx_heat" in tile


def test_reconstruct_budget_dataset_keeps_canonical_shape_with_sparse_velocity():
    tile = cache.build_arco_cache_tile(
        _canonical_dataset(),
        _request(),
        include_benchmark_variables=False,
    )

    out = selection.reconstruct_budget_dataset(tile, _request())

    assert out["u"].dims == ("time", "level", "lat", "lon")
    assert out["v"].dims == ("time", "level", "lat", "lon")
    assert bool(out["u"].sel(lon=13.0).isnull().all())
    assert bool(out["v"].sel(lat=3.0).isnull().all())
    assert bool(out["u"].sel(lon=11.0, lat=[2.0, 3.0, 4.0]).notnull().all())
    assert bool(out["u"].sel(lon=11.0, lat=[1.0, 5.0]).isnull().all())
    assert bool(out["v"].sel(lat=1.0, lon=[12.0, 13.0, 14.0]).notnull().all())
    assert bool(out["v"].sel(lat=1.0, lon=[11.0, 15.0]).isnull().all())


def test_reconstruct_benchmark_dataset_expands_compact_shell():
    tile = cache.build_arco_cache_tile(
        _canonical_dataset(),
        _request(),
        include_benchmark_variables=True,
    )

    out = selection.reconstruct_benchmark_dataset(tile, _request())

    for name in ("Fx_heat", "Fy_heat", "Fx_mass", "Fy_mass"):
        assert out[name].dims == ("time", "lat", "lon")
    for name in variables.COLUMN_BENCHMARK_VAR_NAMES:
        assert out[name].dims == ("time", "lat", "lon")
        xrt.assert_identical(out[name], tile[name])

    assert bool(out["Fx_mass"].sel(lon=11.0, lat=[2.0, 3.0, 4.0]).notnull().all())
    assert bool(out["Fx_mass"].sel(lon=13.0).isnull().all())
    assert bool(out["Fx_mass"].sel(lon=11.0, lat=[1.0, 5.0]).isnull().all())
    assert bool(out["Fy_mass"].sel(lat=1.0, lon=[12.0, 13.0, 14.0]).notnull().all())
    assert bool(out["Fy_mass"].sel(lat=3.0).isnull().all())
    assert bool(out["Fy_mass"].sel(lat=1.0, lon=[11.0, 15.0]).isnull().all())


def test_normalize_zarr_chunks_clears_inherited_arco_encoding():
    tile = cache.build_arco_cache_tile(
        _canonical_dataset().chunk({"time": (2, 3), "lat": (2, 3, 2), "lon": (3, 4)}),
        _request(),
        include_benchmark_variables=True,
    )
    tile["w"].encoding["chunks"] = (1, 37, 721, 1440)
    tile["w"].encoding["preferred_chunks"] = {
        "time": 1,
        "level": 37,
        "lat": 721,
        "lon": 1440,
    }
    tile["w"].encoding["compressor"] = object()
    tile["w"].encoding["filters"] = [object()]

    out = cache._normalize_zarr_chunks(tile, _source_cfg())

    assert "chunks" not in out["w"].encoding
    assert "preferred_chunks" not in out["w"].encoding
    assert "compressor" not in out["w"].encoding
    assert "filters" not in out["w"].encoding
    assert out["w"].chunks == ((5,), (3,), (5,), (5,))
    assert out["Fy_mass"].chunks == ((5,), (4,), (3,))


def test_tile_id_changes_with_vertical_request_fields():
    source_cfg = _source_cfg()
    ids = {
        cache.tile_id_for_request(
            source_cfg,
            _request_with(zg_top_pressure=80000.0, zg_bottom="pressure_level", zg_bottom_pressure=100000.0),
            include_benchmark_variables=False,
        ),
        cache.tile_id_for_request(
            source_cfg,
            _request_with(zg_top_pressure=90000.0, zg_bottom="pressure_level", zg_bottom_pressure=100000.0),
            include_benchmark_variables=False,
        ),
        cache.tile_id_for_request(
            source_cfg,
            _request_with(zg_top_pressure=80000.0, zg_bottom="pressure_level", zg_bottom_pressure=90000.0),
            include_benchmark_variables=False,
        ),
        cache.tile_id_for_request(
            source_cfg,
            _request_with(zg_top_pressure=80000.0, zg_bottom="surface_pressure", zg_bottom_pressure=None),
            include_benchmark_variables=False,
        ),
    }

    assert len(ids) == 4


def test_only_benchmark_tile_id_tracks_benchmark_variable_contract(monkeypatch):
    source_cfg = _source_cfg()
    request = _request()
    core_before = cache.tile_id_for_request(
        source_cfg,
        request,
        include_benchmark_variables=False,
    )
    benchmark_before = cache.tile_id_for_request(
        source_cfg,
        request,
        include_benchmark_variables=True,
    )

    monkeypatch.setattr(
        variables,
        "BENCHMARK_VAR_NAMES",
        (*variables.BENCHMARK_VAR_NAMES, "future_benchmark"),
    )

    assert cache.tile_id_for_request(
        source_cfg,
        request,
        include_benchmark_variables=False,
    ) == core_before
    assert cache.tile_id_for_request(
        source_cfg,
        request,
        include_benchmark_variables=True,
    ) != benchmark_before


def test_write_tile_removes_partial_tmp_store_on_failure(monkeypatch, tmp_path):
    source_cfg = _source_cfg()
    request = _request()
    tile = cache.build_arco_cache_tile(
        _canonical_dataset(),
        request,
        include_benchmark_variables=False,
    )
    tile_id = cache.tile_id_for_request(
        source_cfg,
        request,
        include_benchmark_variables=False,
    )
    tiles_dir = tmp_path / cache.TILES_DIR

    def fake_to_zarr(self, path, mode="w"):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "zarr.json").write_text("{}")
        raise OSError("Temporary failure in name resolution")

    monkeypatch.setattr(xr.Dataset, "to_zarr", fake_to_zarr)

    with pytest.raises(OSError, match="Temporary failure"):
        cache.write_tile(
            tmp_path,
            tile,
            source_cfg,
            request,
            include_benchmark_variables=False,
        )

    assert not list(tiles_dir.glob(f".{tile_id}.*.tmp.zarr"))


def test_write_tile_does_not_hold_cache_lock_during_zarr_write(monkeypatch, tmp_path):
    source_cfg = _source_cfg()
    request = _request()
    tile = cache.build_arco_cache_tile(
        _canonical_dataset(),
        request,
        include_benchmark_variables=False,
    )
    events = []
    original_lock = cache._cache_write_lock

    class SpyLock:
        def __init__(self, *args, **kwargs):
            self._lock = original_lock(*args, **kwargs)

        def __enter__(self):
            events.append("enter")
            return self._lock.__enter__()

        def __exit__(self, exc_type, exc, tb):
            events.append("exit")
            return self._lock.__exit__(exc_type, exc, tb)

    def fake_to_zarr(self, path, mode="w"):
        assert events == []
        Path(path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cache, "_cache_write_lock", SpyLock)
    monkeypatch.setattr(xr.Dataset, "to_zarr", fake_to_zarr)

    tile_path = cache.write_tile(
        tmp_path,
        tile,
        source_cfg,
        request,
        include_benchmark_variables=False,
    )

    assert tile_path.exists()
    assert events == ["enter", "exit"]


def test_temporary_tile_write_requires_parent_commit(monkeypatch, tmp_path):
    source_cfg = _source_cfg()
    request = _request()
    tile = cache.build_arco_cache_tile(
        _canonical_dataset(),
        request,
        include_benchmark_variables=False,
    )
    _patch_to_zarr_creates_store(monkeypatch)
    temporary_path = cache._create_temporary_tile_path(
        tmp_path,
        source_cfg,
        request,
        include_benchmark_variables=False,
    )

    metadata = cache._write_temporary_tile(
        tmp_path,
        temporary_path,
        tile,
        source_cfg,
        request,
        include_benchmark_variables=False,
    )

    assert temporary_path.exists()
    assert not cache.exact_tile_exists(
        tmp_path,
        source_cfg,
        request,
        include_benchmark_variables=False,
    )

    tile_path = cache._commit_temporary_tile(
        tmp_path,
        temporary_path,
        metadata,
        source_cfg,
        request,
        include_benchmark_variables=False,
    )

    assert tile_path.exists()
    assert not temporary_path.exists()
    assert cache.exact_tile_exists(
        tmp_path,
        source_cfg,
        request,
        include_benchmark_variables=False,
    )


def test_parent_commit_removes_concurrent_duplicate_temporary_tile(monkeypatch, tmp_path):
    source_cfg = _source_cfg()
    request = _request()
    tile = cache.build_arco_cache_tile(
        _canonical_dataset(),
        request,
        include_benchmark_variables=False,
    )
    _patch_to_zarr_creates_store(monkeypatch)
    temporary_paths = [
        cache._create_temporary_tile_path(
            tmp_path,
            source_cfg,
            request,
            include_benchmark_variables=False,
        )
        for _ in range(2)
    ]
    metadata = [
        cache._write_temporary_tile(
            tmp_path,
            path,
            tile,
            source_cfg,
            request,
            include_benchmark_variables=False,
        )
        for path in temporary_paths
    ]

    first_path = cache._commit_temporary_tile(
        tmp_path,
        temporary_paths[0],
        metadata[0],
        source_cfg,
        request,
        include_benchmark_variables=False,
    )
    second_path = cache._commit_temporary_tile(
        tmp_path,
        temporary_paths[1],
        metadata[1],
        source_cfg,
        request,
        include_benchmark_variables=False,
    )

    assert first_path == second_path
    assert first_path.exists()
    assert not any(path.exists() for path in temporary_paths)


def test_staged_arco_retrieval_retries_transient_write_failures(monkeypatch, tmp_path):
    source_cfg = _source_cfg()
    request = _request()
    attempt_calls = []
    sleeps = []

    def fake_run_stage_attempt(*args, **kwargs):
        attempt_calls.append((args, kwargs))
        if len(attempt_calls) == 1:
            raise OSError("Temporary failure in name resolution")
        return tmp_path / "tile.zarr", _tile_write_metadata()

    monkeypatch.setattr(staged_arco_retrieval, "_run_stage_attempt_in_child", fake_run_stage_attempt)
    monkeypatch.setattr(staged_arco_retrieval.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(staged_arco_retrieval.config, "DEFAULT_ARCO_OPEN_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(staged_arco_retrieval.config, "DEFAULT_ARCO_OPEN_RETRY_BASE_DELAY_SECONDS", 3.0)

    tile_path, metadata = staged_arco_retrieval._stage_window_with_retry(
        tmp_path,
        source_cfg,
        request,
        include_benchmark_variables=False,
    )

    assert tile_path == tmp_path / "tile.zarr"
    assert metadata == _tile_write_metadata()
    assert len(attempt_calls) == 2
    assert sleeps == [3.0]


def test_staged_arco_retrieval_retries_stage_attempt_timeout(monkeypatch, tmp_path):
    source_cfg = _source_cfg()
    request = _request()
    attempt_calls = []
    sleeps = []

    def fake_run_stage_attempt(*args, **kwargs):
        attempt_calls.append((args, kwargs))
        if len(attempt_calls) == 1:
            raise TimeoutError("ARCO stage/write attempt exceeded 5 seconds")
        return tmp_path / "tile.zarr", _tile_write_metadata()

    monkeypatch.setattr(staged_arco_retrieval, "_run_stage_attempt_in_child", fake_run_stage_attempt)
    monkeypatch.setattr(staged_arco_retrieval.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(staged_arco_retrieval.config, "DEFAULT_ARCO_OPEN_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(staged_arco_retrieval.config, "DEFAULT_ARCO_OPEN_RETRY_BASE_DELAY_SECONDS", 3.0)

    tile_path, metadata = staged_arco_retrieval._stage_window_with_retry(
        tmp_path,
        source_cfg,
        request,
        include_benchmark_variables=False,
        attempt_timeout_seconds=5.0,
    )

    assert tile_path == tmp_path / "tile.zarr"
    assert metadata == _tile_write_metadata()
    assert len(attempt_calls) == 2
    assert sleeps == [3.0]


def test_staged_arco_retrieval_retries_child_classified_transient_error(monkeypatch, tmp_path):
    attempt_calls = []
    sleeps = []

    def fake_run_stage_attempt(*args, **kwargs):
        attempt_calls.append((args, kwargs))
        if len(attempt_calls) == 1:
            raise staged_arco_retrieval._ChildStageAttemptError(
                "ClientConnectorError",
                "cannot connect to host",
                "remote traceback marker",
                transient=True,
            )
        return tmp_path / "tile.zarr", _tile_write_metadata()

    monkeypatch.setattr(staged_arco_retrieval, "_run_stage_attempt_in_child", fake_run_stage_attempt)
    monkeypatch.setattr(staged_arco_retrieval.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(staged_arco_retrieval.config, "DEFAULT_ARCO_OPEN_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(staged_arco_retrieval.config, "DEFAULT_ARCO_OPEN_RETRY_BASE_DELAY_SECONDS", 3.0)

    tile_path, metadata = staged_arco_retrieval._stage_window_with_retry(
        tmp_path,
        _source_cfg(),
        _request(),
        include_benchmark_variables=False,
    )

    assert tile_path == tmp_path / "tile.zarr"
    assert metadata == _tile_write_metadata()
    assert len(attempt_calls) == 2
    assert sleeps == [3.0]


def test_child_timeout_terminates_reaps_and_cleans_attempt(tmp_path):
    context = multiprocessing.get_context("spawn")
    before = {process.pid for process in multiprocessing.active_children()}

    with pytest.raises(TimeoutError, match="exceeded 3 seconds"):
        staged_arco_retrieval._run_stage_attempt_in_child(
            tmp_path,
            _source_cfg(),
            _request(),
            include_benchmark_variables=False,
            timeout_seconds=3.0,
            process_context=context,
            worker=_spawn_timeout_worker,
            child_exit_grace_seconds=0.2,
        )

    marker_path = tmp_path / "timeout-child.pid"
    assert marker_path.exists()
    child_pid = int(marker_path.read_text())
    assert {process.pid for process in multiprocessing.active_children()} == before
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
    assert not list((tmp_path / cache.TILES_DIR).glob(".*.tmp.zarr"))
    assert not (tmp_path / cache.LOCK_DIR).exists()


def test_successful_child_stuck_during_shutdown_is_reaped_and_committed(tmp_path):
    tile_path, metadata = staged_arco_retrieval._run_stage_attempt_in_child(
        tmp_path,
        _source_cfg(),
        _request(),
        include_benchmark_variables=False,
        timeout_seconds=5.0,
        process_context=multiprocessing.get_context("spawn"),
        worker=_spawn_lingering_success_worker,
        child_exit_grace_seconds=0.2,
    )

    child_pid = int((tile_path / "child.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
    assert metadata == _tile_write_metadata()
    assert cache.exact_tile_exists(
        tmp_path,
        _source_cfg(),
        _request(),
        include_benchmark_variables=False,
    )
    assert not list((tmp_path / cache.TILES_DIR).glob(".*.tmp.zarr"))


def test_zero_timeout_still_uses_isolated_child(tmp_path):
    tile_path, metadata = staged_arco_retrieval._run_stage_attempt_in_child(
        tmp_path,
        _source_cfg(),
        _request(),
        include_benchmark_variables=False,
        timeout_seconds=0,
        process_context=multiprocessing.get_context("spawn"),
        worker=_spawn_delayed_success_worker,
        child_exit_grace_seconds=0.2,
    )

    assert tile_path.exists()
    assert metadata == _tile_write_metadata()


def test_unexpected_child_exit_is_terminal_and_cleans_attempt(tmp_path):
    with pytest.raises(staged_arco_retrieval._UnexpectedChildExit, match="without a message"):
        staged_arco_retrieval._run_stage_attempt_in_child(
            tmp_path,
            _source_cfg(),
            _request(),
            include_benchmark_variables=False,
            timeout_seconds=5.0,
            process_context=multiprocessing.get_context("spawn"),
            worker=_spawn_unexpected_exit_worker,
            child_exit_grace_seconds=0.2,
        )

    assert not list((tmp_path / cache.TILES_DIR).glob(".*.tmp.zarr"))
    assert not (tmp_path / cache.LOCK_DIR).exists()


def test_parent_sigterm_reaps_child_and_cleans_attempt(tmp_path):
    marker_path = tmp_path / "timeout-child.pid"

    def terminate_parent_after_child_starts():
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if marker_path.exists():
                os.kill(os.getpid(), signal.SIGTERM)
                return
            time.sleep(0.05)

    signal_thread = threading.Thread(target=terminate_parent_after_child_starts, daemon=True)
    signal_thread.start()

    with pytest.raises(SystemExit) as exc_info:
        staged_arco_retrieval._run_stage_attempt_in_child(
            tmp_path,
            _source_cfg(),
            _request(),
            include_benchmark_variables=False,
            timeout_seconds=30.0,
            process_context=multiprocessing.get_context("spawn"),
            worker=_spawn_timeout_worker,
            child_exit_grace_seconds=0.2,
        )

    signal_thread.join()
    assert exc_info.value.code == 128 + signal.SIGTERM
    child_pid = int(marker_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
    assert not list((tmp_path / cache.TILES_DIR).glob(".*.tmp.zarr"))
    assert not (tmp_path / cache.LOCK_DIR).exists()


def test_terminal_child_error_preserves_remote_context_without_retry(monkeypatch, tmp_path):
    attempts = []

    def fake_run_stage_attempt(*args, **kwargs):
        attempts.append((args, kwargs))
        raise staged_arco_retrieval._ChildStageAttemptError(
            "ValueError",
            "bad staged tile",
            "remote traceback marker",
            transient=False,
        )

    monkeypatch.setattr(staged_arco_retrieval, "_run_stage_attempt_in_child", fake_run_stage_attempt)

    with pytest.raises(staged_arco_retrieval._ChildStageAttemptError) as exc_info:
        staged_arco_retrieval._stage_window_with_retry(
            tmp_path,
            _source_cfg(),
            _request(),
            include_benchmark_variables=False,
        )

    assert len(attempts) == 1
    assert "ValueError: bad staged tile" in str(exc_info.value)
    assert "remote traceback marker" in str(exc_info.value)


def test_stage_attempt_worker_serializes_transient_error(monkeypatch, tmp_path):
    def fail_build(*args, **kwargs):
        raise OSError("Temporary failure in name resolution")

    monkeypatch.setattr(staged_arco_retrieval, "_build_tile_from_arco", fail_build)
    receive_conn, send_conn = multiprocessing.Pipe(duplex=False)

    staged_arco_retrieval._stage_attempt_worker(
        send_conn,
        str(tmp_path),
        str(tmp_path / "unused.tmp.zarr"),
        _source_cfg(),
        _request(),
        False,
    )
    message = receive_conn.recv()
    receive_conn.close()

    assert message["status"] == "error"
    assert message["error_type"] == "OSError"
    assert message["message"] == "Temporary failure in name resolution"
    assert message["transient"] is True
    assert "fail_build" in message["traceback"]


def test_staged_retrieval_chunks_month_windows():
    windows = list(
        staged_arco_retrieval._iter_chunked_time_windows(
            "1940-05-15T12:00:00",
            "1940-07-02T03:00:00",
            "month",
        )
    )

    assert windows == [
        ("1940-05-15T12:00:00", "1940-05-31T23:00:00"),
        ("1940-06-01T00:00:00", "1940-06-30T23:00:00"),
        ("1940-07-01T00:00:00", "1940-07-02T03:00:00"),
    ]


def test_staged_retrieval_chunks_day_windows():
    windows = list(
        staged_arco_retrieval._iter_chunked_time_windows(
            "1940-06-01T12:00:00",
            "1940-06-03T02:00:00",
            "day",
        )
    )

    assert windows == [
        ("1940-06-01T12:00:00", "1940-06-01T23:00:00"),
        ("1940-06-02T00:00:00", "1940-06-02T23:00:00"),
        ("1940-06-03T00:00:00", "1940-06-03T02:00:00"),
    ]


def test_staged_retrieval_avoids_store_wide_dask_graph(monkeypatch):
    canonical = _canonical_dataset()
    source = canonical[["T", "u", "v", "w", "sp"]].rename(
        {
            "T": "temperature",
            "u": "u_component_of_wind",
            "v": "v_component_of_wind",
            "w": "vertical_velocity",
            "sp": "surface_pressure",
        }
    )
    observed = {}
    materialized = []
    original_load = xr.DataArray.load

    def fake_open(source_cfg, *, chunks="auto"):
        observed["open_chunks"] = chunks
        return source

    def fake_standardize(ds, source_cfg, *, rechunk=True):
        observed["rechunk"] = rechunk
        return canonical

    def tracked_load(data_array, **kwargs):
        materialized.append(data_array.name)
        return original_load(data_array, **kwargs)

    monkeypatch.setattr(staged_arco_retrieval.io, "_open_arco_zarr_with_retry", fake_open)
    monkeypatch.setattr(staged_arco_retrieval.io, "standardize_era5_dataset", fake_standardize)
    monkeypatch.setattr(xr.DataArray, "load", tracked_load)

    tile = staged_arco_retrieval._build_tile_from_arco(
        _source_cfg(),
        _request(),
        include_benchmark_variables=False,
    )

    assert observed == {"open_chunks": None, "rechunk": False}
    assert materialized == ["T", "w", "sp", "u_wall", "v_wall"]
    assert tile.sizes["lat"] < canonical.sizes["lat"]
    assert tile.sizes["lon"] < canonical.sizes["lon"]


def test_staged_retrieval_main_stages_each_month_chunk(monkeypatch, tmp_path):
    args = cli.parse_args(
        [
            "--lat-min",
            "1",
            "--lat-max",
            "5",
            "--lon-min",
            "11",
            "--lon-max",
            "15",
            "--time-start",
            "1940-06-01T00:00:00",
            "--time-end",
            "1940-07-01T02:00:00",
            "--zg-top-pa",
            "80000",
            "--zg-bottom",
            "pressure_level",
            "--zg-bottom-pa",
            "100000",
            "--staged-cache-root",
            str(tmp_path),
            "--no-use-surface-variables",
        ]
    )
    args.stage_time_chunk = "month"
    staged_windows = []

    def fake_stage_window_with_retry(
        cache_root,
        source_cfg,
        request,
        *,
        include_benchmark_variables,
        attempt_timeout_seconds,
    ):
        staged_windows.append((source_cfg.time_start, source_cfg.time_end))
        return tmp_path / f"tile-{len(staged_windows)}.zarr", _tile_write_metadata()

    monkeypatch.setattr(staged_arco_retrieval, "parse_args", lambda: args)
    monkeypatch.setattr(
        staged_arco_retrieval,
        "_stage_window_with_retry",
        fake_stage_window_with_retry,
    )

    staged_arco_retrieval.main()

    assert staged_windows == [
        ("1940-06-01T00:00:00", "1940-06-30T23:00:00"),
        ("1940-07-01T00:00:00", "1940-07-01T02:00:00"),
    ]


def test_staged_retrieval_skips_only_exact_existing_tile(monkeypatch, tmp_path):
    _patch_to_zarr_creates_store(monkeypatch)
    source_cfg = DataSourceConfig(
        kind="arco_era5",
        arco_path=staged_arco_retrieval.config.DEFAULT_ARCO_PATH,
        time_start="1940-06-01T00:00:00",
        time_end="1940-06-01T04:00:00",
    )
    broad_request = _request_with(
        zg_top_pressure=70000.0,
        zg_bottom="pressure_level",
        zg_bottom_pressure=100000.0,
    )
    target_request = _request()
    _write_indexed_tile(tmp_path, source_cfg, broad_request)

    args = cli.parse_args(
        [
            "--lat-min",
            "1",
            "--lat-max",
            "5",
            "--lon-min",
            "11",
            "--lon-max",
            "15",
            "--time-start",
            "1940-06-01T00:00:00",
            "--time-end",
            "1940-06-01T04:00:00",
            "--zg-top-pa",
            "80000",
            "--zg-bottom",
            "pressure_level",
            "--zg-bottom-pa",
            "100000",
            "--staged-cache-root",
            str(tmp_path),
            "--no-use-surface-variables",
        ]
    )
    stage_calls = []

    def fake_run_stage_attempt(*args, **kwargs):
        stage_calls.append((args, kwargs))
        tile = cache.build_arco_cache_tile(
            _canonical_dataset(),
            target_request,
            include_benchmark_variables=False,
        )
        tile_path = cache.write_tile(
            tmp_path,
            tile,
            source_cfg,
            target_request,
            include_benchmark_variables=False,
        )
        return tile_path, cache._tile_write_metadata(tile)

    monkeypatch.setattr(staged_arco_retrieval, "parse_args", lambda: args)
    monkeypatch.setattr(staged_arco_retrieval, "_run_stage_attempt_in_child", fake_run_stage_attempt)

    staged_arco_retrieval.main()

    assert len(stage_calls) == 1
    assert cache.exact_tile_exists(
        tmp_path,
        source_cfg,
        target_request,
        include_benchmark_variables=False,
    )

    monkeypatch.setattr(
        staged_arco_retrieval,
        "_run_stage_attempt_in_child",
        lambda *args, **kwargs: pytest.fail("exact tile should have been skipped"),
    )
    staged_arco_retrieval.main()


def test_cache_load_reconstructs_from_local_tile_without_arco(monkeypatch, tmp_path):
    source_cfg = _source_cfg()
    request = _request()
    tile = cache.build_arco_cache_tile(
        _canonical_dataset(),
        request,
        include_benchmark_variables=False,
    )

    def fake_to_zarr(self, path, mode="w"):
        Path(path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(xr.Dataset, "to_zarr", fake_to_zarr)
    tile_path = cache.write_tile(
        tmp_path,
        tile,
        source_cfg,
        request,
        include_benchmark_variables=False,
    )
    monkeypatch.setattr(cache.xr, "open_zarr", lambda *args, **kwargs: tile)

    out = cache.load_cache_dataset(
        tmp_path,
        source_cfg,
        request,
        include_benchmark_variables=False,
    )

    assert tile_path.exists()
    assert "T" in out
    assert "u" in out
    assert bool(out["u"].sel(lon=13.0).isnull().all())


def test_cache_load_prefers_exact_vertical_tile(monkeypatch, tmp_path):
    _patch_to_zarr_creates_store(monkeypatch)
    source_cfg = _source_cfg()
    request = _request()
    broad_request = _request_with(
        zg_top_pressure=70000.0,
        zg_bottom="pressure_level",
        zg_bottom_pressure=100000.0,
    )
    broad_path, broad_tile = _write_indexed_tile(tmp_path, source_cfg, broad_request)
    exact_path, exact_tile = _write_indexed_tile(tmp_path, source_cfg, request)
    registry = {str(broad_path): broad_tile, str(exact_path): exact_tile}
    opened = []
    _patch_open_zarr_from_registry(monkeypatch, registry, opened)

    out = cache.load_cache_dataset(
        tmp_path,
        _staged_source_cfg(staged_cache_root=str(tmp_path)),
        request,
        include_benchmark_variables=False,
    )

    assert opened == [str(exact_path)]
    assert out.sizes["level"] == exact_tile.sizes["level"]


def test_cache_load_rejects_insufficient_vertical_coverage(monkeypatch, tmp_path):
    _patch_to_zarr_creates_store(monkeypatch)
    source_cfg = _source_cfg()
    shallow_request = _request_with(
        zg_top_pressure=90000.0,
        zg_bottom="pressure_level",
        zg_bottom_pressure=100000.0,
    )
    shallow_path, shallow_tile = _write_indexed_tile(tmp_path, source_cfg, shallow_request)
    _patch_open_zarr_from_registry(monkeypatch, {str(shallow_path): shallow_tile})

    with pytest.raises(cache.OfflineCoverageError, match="No staged ARCO cache tiles"):
        cache.load_cache_dataset(
            tmp_path,
            _staged_source_cfg(staged_cache_root=str(tmp_path)),
            _request(),
            include_benchmark_variables=False,
        )


def test_cache_load_rejects_different_arco_source_path(monkeypatch, tmp_path):
    _patch_to_zarr_creates_store(monkeypatch)
    other_source = _source_cfg_with(arco_path="gs://other.zarr")
    tile_path, tile = _write_indexed_tile(tmp_path, other_source, _request())
    _patch_open_zarr_from_registry(monkeypatch, {str(tile_path): tile})

    with pytest.raises(cache.OfflineCoverageError, match="No staged ARCO cache tiles"):
        cache.load_cache_dataset(
            tmp_path,
            _staged_source_cfg(staged_cache_root=str(tmp_path), arco_path="gs://example.zarr"),
            _request(),
            include_benchmark_variables=False,
        )


def test_cache_load_combines_complementary_time_tiles(monkeypatch, tmp_path):
    _patch_to_zarr_creates_store(monkeypatch)
    request = _request()
    first_source = _source_cfg_with(
        time_start="1940-06-01T00:00:00",
        time_end="1940-06-01T02:00:00",
    )
    second_source = _source_cfg_with(
        time_start="1940-06-01T03:00:00",
        time_end="1940-06-01T04:00:00",
    )
    first_path, first_tile = _write_indexed_tile(tmp_path, first_source, request)
    second_path, second_tile = _write_indexed_tile(tmp_path, second_source, request)
    _patch_open_zarr_from_registry(
        monkeypatch,
        {
            str(first_path): first_tile,
            str(second_path): second_tile,
        },
    )

    out = cache.load_cache_dataset(
        tmp_path,
        _staged_source_cfg(staged_cache_root=str(tmp_path)),
        request,
        include_benchmark_variables=False,
    )

    assert out.sizes["time"] == 5
    assert str(out["time"].values[0]) == "1940-06-01T00:00:00.000000000"
    assert str(out["time"].values[-1]) == "1940-06-01T04:00:00.000000000"


def test_cache_load_rejects_gap_between_time_tiles(monkeypatch, tmp_path):
    _patch_to_zarr_creates_store(monkeypatch)
    request = _request()
    first_source = _source_cfg_with(
        time_start="1940-06-01T00:00:00",
        time_end="1940-06-01T01:00:00",
    )
    second_source = _source_cfg_with(
        time_start="1940-06-01T03:00:00",
        time_end="1940-06-01T04:00:00",
    )
    first_path, first_tile = _write_indexed_tile(tmp_path, first_source, request)
    second_path, second_tile = _write_indexed_tile(tmp_path, second_source, request)
    _patch_open_zarr_from_registry(
        monkeypatch,
        {
            str(first_path): first_tile,
            str(second_path): second_tile,
        },
    )

    with pytest.raises(cache.OfflineCoverageError, match="missing 1 requested hourly"):
        cache.load_cache_dataset(
            tmp_path,
            _staged_source_cfg(staged_cache_root=str(tmp_path)),
            request,
            include_benchmark_variables=False,
        )


def test_cache_load_rejects_conflicting_overlapping_time_tiles(monkeypatch, tmp_path):
    _patch_to_zarr_creates_store(monkeypatch)
    request = _request()
    first_source = _source_cfg_with(
        time_start="1940-06-01T00:00:00",
        time_end="1940-06-01T03:00:00",
    )
    second_source = _source_cfg_with(
        time_start="1940-06-01T02:00:00",
        time_end="1940-06-01T04:00:00",
    )
    first_path, first_tile = _write_indexed_tile(tmp_path, first_source, request)
    conflicting_ds = _canonical_dataset().copy(deep=True)
    conflicting_ds["T"] = conflicting_ds["T"] + 1000.0
    second_path, second_tile = _write_indexed_tile(
        tmp_path,
        second_source,
        request,
        ds=conflicting_ds,
    )
    _patch_open_zarr_from_registry(
        monkeypatch,
        {
            str(first_path): first_tile,
            str(second_path): second_tile,
        },
    )

    with pytest.raises(cache.OfflineCoverageError, match="conflict"):
        cache.load_cache_dataset(
            tmp_path,
            _staged_source_cfg(staged_cache_root=str(tmp_path)),
            request,
            include_benchmark_variables=False,
        )


def test_benchmark_cache_load_rejects_nonbenchmark_tile(monkeypatch, tmp_path):
    _patch_to_zarr_creates_store(monkeypatch)
    source_cfg = _source_cfg()
    tile_path, tile = _write_indexed_tile(
        tmp_path,
        source_cfg,
        _request(),
        include_benchmark_variables=False,
    )
    _patch_open_zarr_from_registry(monkeypatch, {str(tile_path): tile})

    with pytest.raises(cache.OfflineCoverageError, match="No staged ARCO cache tiles"):
        cache.load_cache_dataset(
            tmp_path,
            _staged_source_cfg(staged_cache_root=str(tmp_path)),
            _request(),
            include_benchmark_variables=True,
        )


def test_benchmark_cache_load_requires_current_variable_contract(
    monkeypatch,
    tmp_path,
):
    _patch_to_zarr_creates_store(monkeypatch)
    source_cfg = _source_cfg()
    tile_path, tile = _write_indexed_tile(
        tmp_path,
        source_cfg,
        _request(),
        include_benchmark_variables=True,
    )
    old_contract_tile = tile.drop_vars(variables.COLUMN_BENCHMARK_VAR_NAMES)
    _patch_open_zarr_from_registry(
        monkeypatch,
        {str(tile_path): old_contract_tile},
    )

    with pytest.raises(
        cache.OfflineCoverageError,
        match="older variable contract and must be restaged",
    ):
        cache.load_cache_dataset(
            tmp_path,
            _staged_source_cfg(staged_cache_root=str(tmp_path)),
            _request(),
            include_benchmark_variables=True,
        )


def test_wall_only_reconstruction_matches_full_budget(tmp_path):
    request = _request()
    surface_specs = _surface_specs()
    full_ds = _canonical_dataset()
    tile = cache.build_arco_cache_tile(
        full_ds,
        request,
        include_benchmark_variables=False,
    )
    full_subset = selection.select_staging_horizontal_extent(full_ds, request.bbox)
    full_subset = selection.select_staging_vertical_extent(full_subset, request)
    staged_ds = selection.reconstruct_budget_dataset(tile, request)

    full_domain, full_halo, full_spec = grid.determine_domain(full_subset, request)
    staged_domain, staged_halo, staged_spec = grid.determine_domain(staged_ds, request)

    assert full_spec == staged_spec

    full_result = budget.calculate_budget(
        full_domain,
        full_halo,
        full_spec,
        surface_specs,
        integral_diagnostics_flag=True,
        plot_dir=str(tmp_path / "full"),
        plot_flag=False,
    )
    staged_result = budget.calculate_budget(
        staged_domain,
        staged_halo,
        staged_spec,
        surface_specs,
        integral_diagnostics_flag=True,
        plot_dir=str(tmp_path / "staged"),
        plot_flag=False,
    )

    xrt.assert_allclose(staged_result, full_result)


def test_cache_missing_coverage_fails_clearly(tmp_path):
    source_cfg = DataSourceConfig(
        kind="staged_arco_cache",
        staged_cache_root=str(tmp_path),
        time_start="1940-06-01T00:00:00",
        time_end="1940-06-01T04:00:00",
    )

    with pytest.raises(cache.OfflineCoverageError, match="No staged ARCO cache tiles"):
        cache.load_cache_dataset(tmp_path, source_cfg, _request())


def test_staged_arco_retrieval_rejects_surface_variables(monkeypatch, tmp_path):
    args = cli.parse_args(
        [
            "--region",
            "pnw_bartusek",
            "--staged-cache-root",
            str(tmp_path),
            "--use-surface-variables",
        ]
    )
    monkeypatch.setattr(staged_arco_retrieval, "parse_args", lambda: args)

    with pytest.raises(NotImplementedError, match=variables.SURFACE_VARIABLE_ERROR):
        staged_arco_retrieval.main()
