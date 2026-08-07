"""Grid selection helpers for ARCO staging and staged-cache reconstruction."""

from __future__ import annotations

import numpy as np
import xarray as xr

from src import specs
from . import variables


def cell_edges_from_centers(coord: xr.DataArray, name: str) -> np.ndarray:
    values = np.asarray(coord.values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} coordinate must be one-dimensional.")
    if values.size < 2:
        raise ValueError(f"{name} coordinate must contain at least two points.")

    diffs = np.diff(values)
    if not (np.all(diffs > 0.0) or np.all(diffs < 0.0)):
        raise ValueError(f"{name} coordinate must be strictly monotonic.")

    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * diffs[0]
    edges[-1] = values[-1] + 0.5 * diffs[-1]
    return edges


def with_pressure_bounds_from_levels(ds: xr.Dataset) -> xr.Dataset:
    if "p_start" in ds.coords and "p_end" in ds.coords:
        return ds

    level = ds["level"]
    edges = cell_edges_from_centers(level, "level")
    return ds.assign_coords(
        {
            "p_start": ("level", edges[:-1].astype(np.float64)),
            "p_end": ("level", edges[1:].astype(np.float64)),
            "p_mid": ("level", np.asarray(level.values, dtype=float).astype(np.float64)),
        }
    )


def select_staging_horizontal_extent(
    ds: xr.Dataset,
    bbox: tuple[float, float, float, float],
) -> xr.Dataset:
    lat_min, lat_max, lon_min, lon_max = map(float, bbox)
    lat_index = _coord_interval_overlap_indices(ds["lat"], lat_min, lat_max, "lat")
    lon_index = _coord_interval_overlap_indices(ds["lon"], lon_min, lon_max, "lon")
    return ds.isel(lat=lat_index, lon=lon_index)


def select_staging_vertical_extent(
    ds: xr.Dataset,
    request: specs.DomainRequest,
) -> xr.Dataset:
    ds = with_pressure_bounds_from_levels(ds)

    p_top = float(request.zg_top_pressure)
    if request.zg_bottom == "pressure_level":
        if request.zg_bottom_pressure is None:
            raise ValueError("zg_bottom_pressure must be set when zg_bottom='pressure_level'.")
        p_bottom = float(request.zg_bottom_pressure)
    else:
        p_bottom = float(ds["p_start"].max().values)

    p_start = ds["p_start"].astype("float64")
    p_end = ds["p_end"].astype("float64")
    keep = (p_start >= p_top) & (p_end <= p_bottom)
    indices = np.flatnonzero(np.asarray(keep.values, dtype=bool))
    if indices.size == 0:
        raise ValueError(
            "No pressure levels overlap the requested staging interval "
            f"[{p_top}, {p_bottom}] Pa."
        )

    return ds.isel(level=indices)


def select_wall_only_tile(
    ds: xr.Dataset,
    request: specs.DomainRequest,
    *,
    include_benchmark_variables: bool = False,
) -> xr.Dataset:
    """Return a staged tile with full scalar fields and wall-only velocities."""
    indexers = wall_stencil_indices(ds, request)
    domain_lat_indices = indexers["domain_lat"]
    domain_lon_indices = indexers["domain_lon"]
    u_lon_indices = indexers["u_lon"]
    v_lat_indices = indexers["v_lat"]

    tile = xr.Dataset(
        {
            "T": ds["T"],
            "w": ds["w"],
            "sp": ds["sp"],
            "u_wall": ds["u"]
            .isel(lat=domain_lat_indices, lon=u_lon_indices)
            .rename({"lat": "u_lat", "lon": "u_lon"}),
            "v_wall": ds["v"]
            .isel(lat=v_lat_indices, lon=domain_lon_indices)
            .rename({"lat": "v_lat", "lon": "v_lon"}),
        },
        attrs=dict(ds.attrs),
    )

    for coord_name in ("p_start", "p_end", "p_mid"):
        if coord_name in ds.coords:
            tile = tile.assign_coords({coord_name: ds[coord_name]})

    if include_benchmark_variables:
        tile = _add_wall_only_benchmark_variables(
            tile,
            ds,
            domain_lat_indices,
            domain_lon_indices,
            u_lon_indices,
            v_lat_indices,
        )

    tile.attrs.update(
        {
            "ehb_cache_schema": "staged_arco_cache_v2",
            "ehb_velocity_storage": "wall_only",
            "ehb_wall_storage": "compact_lateral_shell",
        }
    )
    return tile


def reconstruct_budget_dataset(tile: xr.Dataset, request: specs.DomainRequest) -> xr.Dataset:
    """Expand wall-only velocity arrays into the canonical budget schema."""
    indexers = wall_stencil_indices(tile, request)
    required_u_lon = tile["lon"].isel(lon=indexers["u_lon"])
    required_v_lat = tile["lat"].isel(lat=indexers["v_lat"])

    if "u_wall" not in tile or "v_wall" not in tile:
        raise ValueError("Staged cache tile is missing required u_wall/v_wall variables.")

    u_wall = _select_u_wall(tile, indexers, required_u_lon)
    v_wall = _select_v_wall(tile, indexers, required_v_lat)

    u = _expand_sparse_wall(u_wall, tile["T"], name="u")
    v = _expand_sparse_wall(v_wall, tile["T"], name="v")

    out = xr.Dataset(
        {
            "T": tile["T"],
            "u": u,
            "v": v,
            "w": tile["w"],
            "sp": tile["sp"],
        },
        attrs=dict(tile.attrs),
    )
    for coord_name in ("p_start", "p_end", "p_mid"):
        if coord_name in tile.coords:
            out = out.assign_coords({coord_name: tile[coord_name]})
    return out


def reconstruct_benchmark_dataset(tile: xr.Dataset, request: specs.DomainRequest) -> xr.Dataset:
    missing = [
        name
        for name in variables.BENCHMARK_VAR_NAMES
        if name not in tile
    ]
    if missing:
        raise ValueError(
            "Staged ARCO cache does not contain benchmark variables required by "
            f"--include-benchmark-variables: {missing}"
        )

    indexers = wall_stencil_indices(tile, request)
    u_lon = tile["lon"].isel(lon=indexers["u_lon"])
    v_lat = tile["lat"].isel(lat=indexers["v_lat"])

    out_vars = {}
    for name in ("Fx_heat", "Fx_mass"):
        wall = _select_x_benchmark_wall(tile, name, indexers, u_lon)
        out_vars[name] = _expand_sparse_wall(wall, tile["sp"], name=name)
    for name in ("Fy_heat", "Fy_mass"):
        wall = _select_y_benchmark_wall(tile, name, indexers, v_lat)
        out_vars[name] = _expand_sparse_wall(wall, tile["sp"], name=name)
    for name in variables.COLUMN_BENCHMARK_VAR_NAMES:
        out_vars[name] = tile[name].transpose("time", "lat", "lon")

    return xr.Dataset(out_vars, attrs=dict(tile.attrs))


def _expand_sparse_wall(
    wall: xr.DataArray,
    template: xr.DataArray,
    *,
    name: str,
) -> xr.DataArray:
    """Expand a compact wall field without multiplying Dask chunks.

    Xarray's outer alignment builds one mask per missing spatial coordinate.
    If the compact wall retains several spatial chunks, combining those masks
    can multiply the task graph. A compact lateral shell is small in space, so
    coalescing only its spatial chunks before alignment keeps the graph bounded
    while preserving time and level parallelism. The expanded result is then
    restored to the template's chunk layout.
    """
    if wall.chunks is not None:
        spatial_chunks = {
            dim: -1
            for dim in ("lat", "lon")
            if dim in wall.dims
        }
        wall = wall.chunk(spatial_chunks)

    empty = xr.full_like(template, np.nan).rename(name)
    expanded = wall.combine_first(empty).transpose(*template.dims).rename(name)

    if expanded.chunks is not None and template.chunks is not None:
        expanded = expanded.chunk(
            {
                dim: template.chunksizes[dim]
                for dim in template.dims
            }
        )
    return expanded


def _select_u_wall(
    tile: xr.Dataset,
    indexers: dict[str, np.ndarray],
    required_u_lon: xr.DataArray,
) -> xr.DataArray:
    required_u_lat = tile["lat"].isel(lat=indexers["domain_lat"])
    wall = tile["u_wall"].sel(u_lon=required_u_lon.values)
    if "u_lat" in wall.dims:
        wall = wall.sel(u_lat=required_u_lat.values).rename({"u_lat": "lat"})
    elif "lat" in wall.dims:
        wall = wall.sel(lat=required_u_lat.values)
    else:
        raise ValueError("u_wall must contain either a compact u_lat dimension or canonical lat dimension.")
    return wall.rename({"u_lon": "lon"})


def _select_v_wall(
    tile: xr.Dataset,
    indexers: dict[str, np.ndarray],
    required_v_lat: xr.DataArray,
) -> xr.DataArray:
    required_v_lon = tile["lon"].isel(lon=indexers["domain_lon"])
    wall = tile["v_wall"].sel(v_lat=required_v_lat.values)
    if "v_lon" in wall.dims:
        wall = wall.sel(v_lon=required_v_lon.values).rename({"v_lon": "lon"})
    elif "lon" in wall.dims:
        wall = wall.sel(lon=required_v_lon.values)
    else:
        raise ValueError("v_wall must contain either a compact v_lon dimension or canonical lon dimension.")
    return wall.rename({"v_lat": "lat"})


def _select_x_benchmark_wall(
    tile: xr.Dataset,
    name: str,
    indexers: dict[str, np.ndarray],
    required_u_lon: xr.DataArray,
) -> xr.DataArray:
    required_u_lat = tile["lat"].isel(lat=indexers["domain_lat"])
    wall = tile[name]
    if "u_lon" in wall.dims:
        wall = wall.sel(u_lon=required_u_lon.values).rename({"u_lon": "lon"})
    elif "lon" in wall.dims:
        wall = wall.sel(lon=required_u_lon.values)
    else:
        raise ValueError(f"{name} must contain either a compact u_lon dimension or canonical lon dimension.")

    if "u_lat" in wall.dims:
        wall = wall.sel(u_lat=required_u_lat.values).rename({"u_lat": "lat"})
    elif "lat" in wall.dims:
        wall = wall.sel(lat=required_u_lat.values)
    else:
        raise ValueError(f"{name} must contain either a compact u_lat dimension or canonical lat dimension.")
    return wall


def _select_y_benchmark_wall(
    tile: xr.Dataset,
    name: str,
    indexers: dict[str, np.ndarray],
    required_v_lat: xr.DataArray,
) -> xr.DataArray:
    required_v_lon = tile["lon"].isel(lon=indexers["domain_lon"])
    wall = tile[name]
    if "v_lat" in wall.dims:
        wall = wall.sel(v_lat=required_v_lat.values).rename({"v_lat": "lat"})
    elif "lat" in wall.dims:
        wall = wall.sel(lat=required_v_lat.values)
    else:
        raise ValueError(f"{name} must contain either a compact v_lat dimension or canonical lat dimension.")

    if "v_lon" in wall.dims:
        wall = wall.sel(v_lon=required_v_lon.values).rename({"v_lon": "lon"})
    elif "lon" in wall.dims:
        wall = wall.sel(lon=required_v_lon.values)
    else:
        raise ValueError(f"{name} must contain either a compact v_lon dimension or canonical lon dimension.")
    return wall


def wall_stencil_indices(ds: xr.Dataset, request: specs.DomainRequest) -> dict[str, np.ndarray]:
    lat0, lat1, lon0, lon1 = map(float, request.bbox)
    margin = int(request.margin_n)
    if margin < 1:
        raise ValueError("margin_n must be >= 1.")

    lat_vals = np.asarray(ds["lat"].values, dtype=float)
    lon_vals = np.asarray(ds["lon"].values, dtype=float)

    lat_start, lat_stop = _center_index_span(lat_vals, lat0, lat1, "lat")
    lon_start, lon_stop = _center_index_span(lon_vals, lon0, lon1, "lon")

    halo_offset = margin - 1
    halo_lat_start = lat_start + halo_offset
    halo_lat_stop = lat_stop - halo_offset
    halo_lon_start = lon_start + halo_offset
    halo_lon_stop = lon_stop - halo_offset

    if halo_lat_stop - halo_lat_start < 2 or halo_lon_stop - halo_lon_start < 2:
        raise ValueError("Staged domain is too small to build wall velocity stencils.")
    domain_lat_start = lat_start + margin
    domain_lat_stop = lat_stop - margin
    domain_lon_start = lon_start + margin
    domain_lon_stop = lon_stop - margin
    if domain_lat_stop <= domain_lat_start or domain_lon_stop <= domain_lon_start:
        raise ValueError("Staged domain is too small after applying the requested margin.")

    u_lon = np.array(
        [halo_lon_start, halo_lon_start + 1, halo_lon_stop - 2, halo_lon_stop - 1],
        dtype=int,
    )
    v_lat = np.array(
        [halo_lat_start, halo_lat_start + 1, halo_lat_stop - 2, halo_lat_stop - 1],
        dtype=int,
    )
    return {
        "domain_lat": np.arange(domain_lat_start, domain_lat_stop, dtype=int),
        "domain_lon": np.arange(domain_lon_start, domain_lon_stop, dtype=int),
        "u_lon": np.unique(u_lon),
        "v_lat": np.unique(v_lat),
    }


def _add_wall_only_benchmark_variables(
    tile: xr.Dataset,
    ds: xr.Dataset,
    domain_lat_indices: np.ndarray,
    domain_lon_indices: np.ndarray,
    u_lon_indices: np.ndarray,
    v_lat_indices: np.ndarray,
) -> xr.Dataset:
    for name in ("Fx_heat", "Fx_mass"):
        if name not in ds:
            raise ValueError(f"Benchmark variable {name!r} is missing from staged source dataset.")
        tile[name] = (
            ds[name]
            .isel(lat=domain_lat_indices, lon=u_lon_indices)
            .rename({"lat": "u_lat", "lon": "u_lon"})
        )

    for name in ("Fy_heat", "Fy_mass"):
        if name not in ds:
            raise ValueError(f"Benchmark variable {name!r} is missing from staged source dataset.")
        tile[name] = (
            ds[name]
            .isel(lat=v_lat_indices, lon=domain_lon_indices)
            .rename({"lat": "v_lat", "lon": "v_lon"})
        )

    for name in variables.COLUMN_BENCHMARK_VAR_NAMES:
        if name not in ds:
            raise ValueError(f"Benchmark variable {name!r} is missing from staged source dataset.")
        tile[name] = ds[name]

    return tile


def _coord_interval_overlap_indices(
    coord: xr.DataArray,
    lower: float,
    upper: float,
    name: str,
) -> np.ndarray:
    edges = cell_edges_from_centers(coord, name)
    starts = np.minimum(edges[:-1], edges[1:])
    ends = np.maximum(edges[:-1], edges[1:])
    keep = (ends >= lower) & (starts <= upper)
    indices = np.flatnonzero(keep)
    if indices.size == 0:
        raise ValueError(f"No {name} cells overlap requested interval [{lower}, {upper}].")
    return indices


def _center_index_span(
    values: np.ndarray,
    lower: float,
    upper: float,
    name: str,
) -> tuple[int, int]:
    mask = (values >= lower) & (values <= upper)
    if not np.any(mask):
        raise ValueError(f"No {name} centers fall inside requested interval [{lower}, {upper}].")
    return int(np.flatnonzero(mask)[0]), int(np.flatnonzero(mask)[-1]) + 1
