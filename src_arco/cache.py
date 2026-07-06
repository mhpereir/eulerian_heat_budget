"""Indexed local Zarr cache for staged ARCO ERA5 budget inputs."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np
import xarray as xr

from src import config, specs
from . import selection


DB_NAME = "cache.sqlite"
TILES_DIR = "tiles"
LOCK_DIR = ".write.lock"
CACHE_SCHEMA = "staged_arco_cache_v2"


class OfflineCoverageError(RuntimeError):
    """Raised when a staged ARCO cache cannot satisfy an offline request."""


def init_cache_root(cache_root: str | Path) -> Path:
    root = Path(cache_root)
    (root / TILES_DIR).mkdir(parents=True, exist_ok=True)
    with _connect(root) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tiles (
                tile_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                time_start TEXT,
                time_end TEXT,
                lat_min REAL,
                lat_max REAL,
                lon_min REAL,
                lon_max REAL,
                level_min REAL,
                level_max REAL,
                include_benchmark INTEGER NOT NULL,
                request_json TEXT NOT NULL,
                source_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
    return root


def tile_id_for_request(
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
    *,
    include_benchmark_variables: bool,
) -> str:
    payload = {
        "source": {
            "kind": source_cfg.kind,
            "arco_path": source_cfg.arco_path,
            "time_start": source_cfg.time_start,
            "time_end": source_cfg.time_end,
        },
        "request": asdict(request),
        "include_benchmark_variables": bool(include_benchmark_variables),
        "schema": CACHE_SCHEMA,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def write_tile(
    cache_root: str | Path,
    tile: xr.Dataset,
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
    *,
    include_benchmark_variables: bool,
) -> Path:
    root = init_cache_root(cache_root)
    tile_id = tile_id_for_request(
        source_cfg,
        request,
        include_benchmark_variables=include_benchmark_variables,
    )
    rel_path = Path(TILES_DIR) / f"{tile_id}.zarr"
    tile_path = root / rel_path

    with _cache_write_lock(root):
        if tile_path.exists():
            _register_tile(root, tile_id, rel_path, tile, source_cfg, request, include_benchmark_variables)
            return tile_path

        tmp_path = root / TILES_DIR / f".{tile_id}.tmp.zarr"
        tile_to_write = _normalize_zarr_chunks(tile, source_cfg)
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        try:
            tile_to_write.to_zarr(str(tmp_path), mode="w")
        except Exception:
            if tmp_path.exists():
                shutil.rmtree(tmp_path)
            raise
        tmp_path.replace(tile_path)
        _register_tile(root, tile_id, rel_path, tile_to_write, source_cfg, request, include_benchmark_variables)
    return tile_path


def cache_has_coverage(
    cache_root: str | Path,
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
    *,
    include_benchmark_variables: bool = False,
) -> bool:
    try:
        load_cache_dataset(
            cache_root,
            source_cfg,
            request,
            include_benchmark_variables=include_benchmark_variables,
        )
    except OfflineCoverageError:
        return False
    return True


def load_cache_dataset(
    cache_root: str | Path,
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
    *,
    include_benchmark_variables: bool = False,
) -> xr.Dataset:
    root = init_cache_root(cache_root)
    candidates = _candidate_tile_paths(root, source_cfg, request, include_benchmark_variables)
    if not candidates:
        raise OfflineCoverageError(
            "No staged ARCO cache tiles overlap the requested time/domain coverage."
        )

    combined: xr.Dataset | None = None
    for path in candidates:
        ds = xr.open_zarr(str(path), decode_timedelta=False)
        combined = ds if combined is None else combined.combine_first(ds)

    if combined is None:
        raise OfflineCoverageError("No staged ARCO cache tiles could be opened.")

    selected = _select_cached_coverage(combined, source_cfg, request)
    _validate_wall_coverage(selected, request)

    if include_benchmark_variables:
        return selection.reconstruct_benchmark_dataset(selected, request)
    return selection.reconstruct_budget_dataset(selected, request)


def build_arco_cache_tile(
    ds: xr.Dataset,
    request: specs.DomainRequest,
    *,
    include_benchmark_variables: bool,
) -> xr.Dataset:
    ds = selection.select_staging_horizontal_extent(ds, request.bbox)
    ds = selection.select_staging_vertical_extent(ds, request)
    return selection.select_wall_only_tile(
        ds,
        request,
        include_benchmark_variables=include_benchmark_variables,
    )


def _select_cached_coverage(
    ds: xr.Dataset,
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
) -> xr.Dataset:
    if source_cfg.time_start is not None or source_cfg.time_end is not None:
        ds = ds.sel(time=slice(source_cfg.time_start, source_cfg.time_end))
    if ds.sizes.get("time", 0) == 0:
        raise OfflineCoverageError("Staged ARCO cache does not cover the requested time window.")

    try:
        ds = selection.select_staging_horizontal_extent(ds, request.bbox)
        ds = selection.select_staging_vertical_extent(ds, request)
    except ValueError as exc:
        raise OfflineCoverageError(str(exc)) from exc

    _require_time_bounds(ds, source_cfg)
    return ds


def _validate_wall_coverage(ds: xr.Dataset, request: specs.DomainRequest) -> None:
    try:
        indexers = selection.wall_stencil_indices(ds, request)
    except ValueError as exc:
        raise OfflineCoverageError(str(exc)) from exc

    required_u_lon = set(np.asarray(ds["lon"].isel(lon=indexers["u_lon"]).values, dtype=float))
    required_v_lat = set(np.asarray(ds["lat"].isel(lat=indexers["v_lat"]).values, dtype=float))
    required_u_lat = set(np.asarray(ds["lat"].isel(lat=indexers["domain_lat"]).values, dtype=float))
    required_v_lon = set(np.asarray(ds["lon"].isel(lon=indexers["domain_lon"]).values, dtype=float))
    available_u_lon = set(np.asarray(ds["u_wall"]["u_lon"].values, dtype=float)) if "u_wall" in ds else set()
    available_v_lat = set(np.asarray(ds["v_wall"]["v_lat"].values, dtype=float)) if "v_wall" in ds else set()
    available_u_lat = _available_wall_coord_values(ds, "u_wall", compact_coord="u_lat", canonical_coord="lat")
    available_v_lon = _available_wall_coord_values(ds, "v_wall", compact_coord="v_lon", canonical_coord="lon")

    if not required_u_lon.issubset(available_u_lon):
        missing = sorted(required_u_lon - available_u_lon)
        raise OfflineCoverageError(f"Staged ARCO cache missing u_wall lon stencils: {missing}")
    if not required_v_lat.issubset(available_v_lat):
        missing = sorted(required_v_lat - available_v_lat)
        raise OfflineCoverageError(f"Staged ARCO cache missing v_wall lat stencils: {missing}")
    if not required_u_lat.issubset(available_u_lat):
        missing = sorted(required_u_lat - available_u_lat)
        raise OfflineCoverageError(f"Staged ARCO cache missing u_wall domain lat coverage: {missing}")
    if not required_v_lon.issubset(available_v_lon):
        missing = sorted(required_v_lon - available_v_lon)
        raise OfflineCoverageError(f"Staged ARCO cache missing v_wall domain lon coverage: {missing}")


def _available_wall_coord_values(
    ds: xr.Dataset,
    variable: str,
    *,
    compact_coord: str,
    canonical_coord: str,
) -> set[float]:
    if variable not in ds:
        return set()
    da = ds[variable]
    if compact_coord in da.coords:
        values = da[compact_coord].values
    elif canonical_coord in da.coords:
        values = da[canonical_coord].values
    else:
        return set()
    return set(np.asarray(values, dtype=float))


def _normalize_zarr_chunks(tile: xr.Dataset, source_cfg: specs.DataSourceConfig) -> xr.Dataset:
    chunk_spec: dict[str, int] = {}
    for dim, size in tile.sizes.items():
        if dim == "time":
            chunk_spec[dim] = max(1, min(int(size), int(source_cfg.chunks_time)))
        elif dim == "level":
            chunk_spec[dim] = int(size)
        elif dim in {"lat", "lon", "u_lat", "v_lat", "u_lon", "v_lon"}:
            base = config.n_lat if dim in {"lat", "u_lat", "v_lat"} else config.n_lon
            chunk_spec[dim] = max(1, min(int(size), int(base)))
        else:
            chunk_spec[dim] = int(size)
    rechunked = tile.chunk(chunk_spec)
    return _clear_inherited_zarr_chunk_encoding(rechunked)


def _clear_inherited_zarr_chunk_encoding(tile: xr.Dataset) -> xr.Dataset:
    tile = tile.copy()
    for variable in tile.variables.values():
        variable.encoding.clear()
    return tile


def _require_time_bounds(ds: xr.Dataset, source_cfg: specs.DataSourceConfig) -> None:
    if source_cfg.time_start is None and source_cfg.time_end is None:
        return

    time_values = ds["time"].values
    if time_values.size == 0:
        raise OfflineCoverageError("Staged ARCO cache does not contain requested times.")

    if source_cfg.time_start is not None:
        start = np.datetime64(source_cfg.time_start)
        if np.asarray(time_values).min() > start:
            raise OfflineCoverageError(
                f"Staged ARCO cache starts after requested time_start={source_cfg.time_start}."
            )
    if source_cfg.time_end is not None:
        end = np.datetime64(source_cfg.time_end)
        if np.asarray(time_values).max() < end:
            raise OfflineCoverageError(
                f"Staged ARCO cache ends before requested time_end={source_cfg.time_end}."
            )


def _candidate_tile_paths(
    root: Path,
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
    include_benchmark_variables: bool,
) -> list[Path]:
    init_cache_root(root)
    with _connect(root) as conn:
        rows = conn.execute(
            """
            SELECT path FROM tiles
            WHERE include_benchmark >= ?
              AND (time_end IS NULL OR ? IS NULL OR time_end >= ?)
              AND (time_start IS NULL OR ? IS NULL OR time_start <= ?)
              AND lat_max >= ?
              AND lat_min <= ?
              AND lon_max >= ?
              AND lon_min <= ?
            ORDER BY created_at ASC
            """,
            (
                1 if include_benchmark_variables else 0,
                source_cfg.time_start,
                source_cfg.time_start,
                source_cfg.time_end,
                source_cfg.time_end,
                float(request.bbox[0]),
                float(request.bbox[1]),
                float(request.bbox[2]),
                float(request.bbox[3]),
            ),
        ).fetchall()

    return [root / row[0] for row in rows if (root / row[0]).exists()]


def _register_tile(
    root: Path,
    tile_id: str,
    rel_path: Path,
    tile: xr.Dataset,
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
    include_benchmark_variables: bool,
) -> None:
    with _connect(root) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO tiles (
                tile_id, path, time_start, time_end, lat_min, lat_max,
                lon_min, lon_max, level_min, level_max, include_benchmark,
                request_json, source_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tile_id,
                str(rel_path),
                _coord_min_as_str(tile, "time"),
                _coord_max_as_str(tile, "time"),
                float(tile["lat"].min().values),
                float(tile["lat"].max().values),
                float(tile["lon"].min().values),
                float(tile["lon"].max().values),
                float(tile["level"].min().values),
                float(tile["level"].max().values),
                1 if include_benchmark_variables else 0,
                json.dumps(asdict(request), sort_keys=True),
                json.dumps(asdict(source_cfg), sort_keys=True),
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def _coord_min_as_str(ds: xr.Dataset, coord: str) -> str | None:
    if coord not in ds.coords or ds[coord].size == 0:
        return None
    return str(ds[coord].min().values)


def _coord_max_as_str(ds: xr.Dataset, coord: str) -> str | None:
    if coord not in ds.coords or ds[coord].size == 0:
        return None
    return str(ds[coord].max().values)


def _connect(root: Path) -> sqlite3.Connection:
    root.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(root / DB_NAME)


class _cache_write_lock:
    def __init__(self, root: Path, *, timeout_seconds: float = 300.0, poll_seconds: float = 1.0):
        self.lock_dir = root / LOCK_DIR
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds

    def __enter__(self):
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self.lock_dir.mkdir()
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for staged cache write lock: {self.lock_dir}")
                time.sleep(self.poll_seconds)

    def __exit__(self, exc_type, exc, tb):
        try:
            self.lock_dir.rmdir()
        finally:
            return False
