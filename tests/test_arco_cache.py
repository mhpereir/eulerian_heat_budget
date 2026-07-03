import importlib
import sys
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
    assert tile["u_wall"].dims == ("time", "level", "lat", "u_lon")
    assert tile["v_wall"].dims == ("time", "level", "v_lat", "lon")
    assert set(tile["u_wall"]["u_lon"].values) == {11.0, 12.0, 14.0, 15.0}
    assert set(tile["v_wall"]["v_lat"].values) == {1.0, 2.0, 4.0, 5.0}
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
    assert bool(out["u"].sel(lon=11.0).notnull().all())
    assert bool(out["v"].sel(lat=1.0).notnull().all())


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
