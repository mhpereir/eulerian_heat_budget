"""Indexed local Zarr cache for staged ARCO ERA5 budget inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
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


def _log_cache_write(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[info] {timestamp} staged cache: {message}", flush=True)


@dataclass(frozen=True)
class _TileRecord:
    tile_id: str
    path: Path
    time_start: str | None
    time_end: str | None
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    level_min: float
    level_max: float
    include_benchmark: bool
    request: dict
    source: dict
    created_at: str


@dataclass(frozen=True)
class _TileWriteMetadata:
    sizes: tuple[tuple[str, int], ...]
    variables: tuple[str, ...]
    time_start: str | None
    time_end: str | None
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    level_min: float
    level_max: float


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


def exact_tile_path(
    cache_root: str | Path,
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
    *,
    include_benchmark_variables: bool,
) -> Path:
    tile_id = tile_id_for_request(
        source_cfg,
        request,
        include_benchmark_variables=include_benchmark_variables,
    )
    return Path(cache_root) / TILES_DIR / f"{tile_id}.zarr"


def exact_tile_exists(
    cache_root: str | Path,
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
    *,
    include_benchmark_variables: bool,
) -> bool:
    root = init_cache_root(cache_root)
    tile_id = tile_id_for_request(
        source_cfg,
        request,
        include_benchmark_variables=include_benchmark_variables,
    )
    with _connect(root) as conn:
        row = conn.execute(
            "SELECT path FROM tiles WHERE tile_id = ?",
            (tile_id,),
        ).fetchone()
    return row is not None and (root / row[0]).exists()


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

    if tile_path.exists():
        with _cache_write_lock(root):
            if tile_path.exists():
                _register_tile(
                    root,
                    tile_id,
                    rel_path,
                    _tile_write_metadata(tile),
                    source_cfg,
                    request,
                    include_benchmark_variables,
                )
                return tile_path

    tmp_path = _new_tmp_tile_path(root, tile_id)
    try:
        metadata = _write_temporary_tile(
            root,
            tmp_path,
            tile,
            source_cfg,
            request,
            include_benchmark_variables=include_benchmark_variables,
        )
        return _commit_temporary_tile(
            root,
            tmp_path,
            metadata,
            source_cfg,
            request,
            include_benchmark_variables=include_benchmark_variables,
        )
    except Exception as exc:
        _remove_temporary_tile(
            tmp_path,
            reason=f"after {type(exc).__name__}: {exc}",
        )
        raise


def _create_temporary_tile_path(
    cache_root: str | Path,
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
    return _new_tmp_tile_path(root, tile_id)


def _write_temporary_tile(
    cache_root: str | Path,
    tmp_path: str | Path,
    tile: xr.Dataset,
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
    *,
    include_benchmark_variables: bool,
) -> _TileWriteMetadata:
    root = Path(cache_root)
    tile_id = tile_id_for_request(
        source_cfg,
        request,
        include_benchmark_variables=include_benchmark_variables,
    )
    tmp_path = _validated_tmp_tile_path(root, tmp_path, tile_id)

    _log_cache_write(f"normalizing zarr chunks for tile {tile_id}")
    tile_to_write = _normalize_zarr_chunks(tile, source_cfg)
    metadata = _tile_write_metadata(tile_to_write)
    write_started = time.monotonic()
    _log_cache_write(f"writing temporary zarr store {tmp_path}")
    tile_to_write.to_zarr(str(tmp_path), mode="w")
    _log_cache_write(
        f"finished temporary zarr store {tmp_path} "
        f"in {time.monotonic() - write_started:.1f}s"
    )
    return metadata


def _commit_temporary_tile(
    cache_root: str | Path,
    tmp_path: str | Path,
    metadata: _TileWriteMetadata,
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
    tmp_path = _validated_tmp_tile_path(root, tmp_path, tile_id)
    rel_path = Path(TILES_DIR) / f"{tile_id}.zarr"
    tile_path = root / rel_path

    with _cache_write_lock(root):
        if tile_path.exists():
            _log_cache_write(f"tile {tile_id} already exists; removing temporary zarr store")
            shutil.rmtree(tmp_path)
        else:
            if not tmp_path.exists():
                raise FileNotFoundError(f"Temporary staged tile does not exist: {tmp_path}")
            _log_cache_write(f"promoting temporary zarr store to {tile_path}")
            tmp_path.replace(tile_path)
        _register_tile(
            root,
            tile_id,
            rel_path,
            metadata,
            source_cfg,
            request,
            include_benchmark_variables,
        )
        _log_cache_write(f"registered tile {tile_id}")
    return tile_path


def _remove_temporary_tile(tmp_path: str | Path, *, reason: str) -> None:
    path = Path(tmp_path)
    if not path.exists():
        return
    _log_cache_write(f"removing temporary zarr store {path} {reason}")
    shutil.rmtree(path)


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
    cache_source_cfg = _cache_source_cfg_for_lookup(source_cfg)
    candidates = _candidate_tile_records(root, cache_source_cfg, request, include_benchmark_variables)
    selected_records = _select_tile_records_for_request(
        cache_source_cfg,
        request,
        candidates,
        include_benchmark_variables,
    )
    if not selected_records:
        raise OfflineCoverageError(
            "No staged ARCO cache tiles cover the requested source/time/domain coverage."
        )

    selected_tiles = [
        _select_cached_coverage(
            xr.open_zarr(str(record.path), decode_timedelta=False),
            cache_source_cfg,
            request,
            require_time_bounds=False,
        )
        for record in selected_records
    ]
    combined = _combine_time_tiles(selected_tiles)

    if combined is None:
        raise OfflineCoverageError("No staged ARCO cache tiles could be opened.")

    _require_time_coverage(combined, cache_source_cfg)
    _validate_wall_coverage(combined, request)

    if include_benchmark_variables:
        return selection.reconstruct_benchmark_dataset(combined, request)
    return selection.reconstruct_budget_dataset(combined, request)


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
    *,
    require_time_bounds: bool = True,
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

    if require_time_bounds:
        _require_time_coverage(ds, source_cfg)
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


def _require_time_coverage(ds: xr.Dataset, source_cfg: specs.DataSourceConfig) -> None:
    _require_time_bounds(ds, source_cfg)

    if source_cfg.time_start is None or source_cfg.time_end is None:
        return

    start = np.datetime64(source_cfg.time_start, "ns")
    end = np.datetime64(source_cfg.time_end, "ns")
    hour = np.timedelta64(1, "h")
    if end < start:
        raise OfflineCoverageError(
            f"Requested time_end={source_cfg.time_end} is before time_start={source_cfg.time_start}."
        )
    if (end - start) % hour != np.timedelta64(0, "ns"):
        raise OfflineCoverageError(
            "Staged ARCO cache coverage checks require hourly-aligned requested time bounds."
        )

    available = np.unique(np.asarray(ds["time"].values).astype("datetime64[ns]"))
    expected = np.arange(start, end + hour, hour, dtype="datetime64[ns]")
    missing = np.setdiff1d(expected, available, assume_unique=True)
    if missing.size:
        preview = ", ".join(str(value) for value in missing[:5])
        if missing.size > 5:
            preview = f"{preview}, ..."
        raise OfflineCoverageError(
            f"Staged ARCO cache is missing {missing.size} requested hourly time(s): {preview}"
        )


def _cache_source_cfg_for_lookup(source_cfg: specs.DataSourceConfig) -> specs.DataSourceConfig:
    if source_cfg.kind == "staged_arco_cache":
        return specs.DataSourceConfig(
            kind="arco_era5",
            arco_path=source_cfg.arco_path or config.DEFAULT_ARCO_PATH,
            arco_storage_token=source_cfg.arco_storage_token,
            chunks_time=source_cfg.chunks_time,
            time_start=source_cfg.time_start,
            time_end=source_cfg.time_end,
        )
    return source_cfg


def _candidate_tile_records(
    root: Path,
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
    include_benchmark_variables: bool,
) -> list[_TileRecord]:
    init_cache_root(root)
    with _connect(root) as conn:
        rows = conn.execute(
            """
            SELECT tile_id, path, time_start, time_end, lat_min, lat_max,
                   lon_min, lon_max, level_min, level_max, include_benchmark,
                   request_json, source_json, created_at
            FROM tiles
            ORDER BY created_at ASC
            """
        ).fetchall()

    records = [_tile_record_from_row(root, row) for row in rows]
    return [
        record
        for record in records
        if record.path.exists()
        and _record_source_matches(record, source_cfg)
        and _record_benchmark_matches(record, include_benchmark_variables)
        and _record_horizontal_covers(record, request)
        and _record_vertical_covers(record, request)
        and _record_time_overlaps(record, source_cfg)
    ]


def _tile_record_from_row(root: Path, row) -> _TileRecord:
    return _TileRecord(
        tile_id=str(row[0]),
        path=root / row[1],
        time_start=row[2],
        time_end=row[3],
        lat_min=float(row[4]),
        lat_max=float(row[5]),
        lon_min=float(row[6]),
        lon_max=float(row[7]),
        level_min=float(row[8]),
        level_max=float(row[9]),
        include_benchmark=bool(row[10]),
        request=json.loads(row[11]),
        source=json.loads(row[12]),
        created_at=str(row[13]),
    )


def _select_tile_records_for_request(
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
    candidates: list[_TileRecord],
    include_benchmark_variables: bool,
) -> list[_TileRecord]:
    exact_id = tile_id_for_request(
        source_cfg,
        request,
        include_benchmark_variables=include_benchmark_variables,
    )
    exact = [record for record in candidates if record.tile_id == exact_id]
    if exact:
        return [_sort_tile_records(exact, request, include_benchmark_variables)[0]]

    full_cover = [record for record in candidates if _record_time_covers(record, source_cfg)]
    if full_cover:
        return [_sort_tile_records(full_cover, request, include_benchmark_variables)[0]]

    return _select_time_mosaic_records(
        candidates,
        source_cfg,
        request,
        include_benchmark_variables,
    )


def _sort_tile_records(
    records: list[_TileRecord],
    request: specs.DomainRequest,
    include_benchmark_variables: bool,
) -> list[_TileRecord]:
    return sorted(
        records,
        key=lambda record: (
            _benchmark_sort_penalty(record, include_benchmark_variables),
            0 if _record_vertical_request_matches(record, request) else 1,
            _record_vertical_span(record),
            _record_horizontal_area(record),
            _record_time_span(record),
            -_created_at_timestamp(record.created_at),
            record.tile_id,
        ),
    )


def _select_time_mosaic_records(
    candidates: list[_TileRecord],
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
    include_benchmark_variables: bool,
) -> list[_TileRecord]:
    benchmark_classes = [False, True] if not include_benchmark_variables else [True]
    for include_benchmark in benchmark_classes:
        selected: list[_TileRecord] = []
        covered_end: np.datetime64 | None = None
        class_candidates = [
            record
            for record in candidates
            if record.include_benchmark == include_benchmark
        ]
        time_sorted = sorted(
            class_candidates,
            key=lambda record: (
                _datetime_sort_value(_to_datetime64(record.time_start), none_first=True),
                _datetime_sort_value(_to_datetime64(record.time_end), none_first=False),
                _record_vertical_span(record),
                _record_horizontal_area(record),
                record.tile_id,
            ),
        )
        for record in time_sorted:
            record_end = _to_datetime64(record.time_end)
            if covered_end is not None and record_end is not None and record_end <= covered_end:
                continue
            selected.append(record)
            if record_end is not None:
                covered_end = record_end if covered_end is None else max(covered_end, record_end)

        if selected and _records_time_bound_request(selected, source_cfg):
            return selected
    return []


def _record_source_matches(record: _TileRecord, source_cfg: specs.DataSourceConfig) -> bool:
    return (
        record.source.get("kind") == source_cfg.kind
        and record.source.get("arco_path") == source_cfg.arco_path
    )


def _record_benchmark_matches(record: _TileRecord, include_benchmark_variables: bool) -> bool:
    if include_benchmark_variables:
        return record.include_benchmark
    return True


def _record_horizontal_covers(record: _TileRecord, request: specs.DomainRequest) -> bool:
    lat_min, lat_max, lon_min, lon_max = map(float, request.bbox)
    return (
        record.lat_min <= lat_min
        and record.lat_max >= lat_max
        and record.lon_min <= lon_min
        and record.lon_max >= lon_max
    )


def _record_vertical_covers(record: _TileRecord, request: specs.DomainRequest) -> bool:
    p_top = float(request.zg_top_pressure)
    if request.zg_bottom == "pressure_level":
        if request.zg_bottom_pressure is None:
            return False
        return record.level_min <= p_top and record.level_max >= float(request.zg_bottom_pressure)
    return record.request.get("zg_bottom") == "surface_pressure" and record.level_min <= p_top


def _record_vertical_request_matches(record: _TileRecord, request: specs.DomainRequest) -> bool:
    return (
        float(record.request.get("zg_top_pressure")) == float(request.zg_top_pressure)
        and record.request.get("zg_bottom") == request.zg_bottom
        and _optional_float_equal(
            record.request.get("zg_bottom_pressure"),
            request.zg_bottom_pressure,
        )
    )


def _record_time_overlaps(record: _TileRecord, source_cfg: specs.DataSourceConfig) -> bool:
    request_start = _to_datetime64(source_cfg.time_start)
    request_end = _to_datetime64(source_cfg.time_end)
    record_start = _to_datetime64(record.time_start)
    record_end = _to_datetime64(record.time_end)
    return (
        (record_end is None or request_start is None or record_end >= request_start)
        and (record_start is None or request_end is None or record_start <= request_end)
    )


def _record_time_covers(record: _TileRecord, source_cfg: specs.DataSourceConfig) -> bool:
    request_start = _to_datetime64(source_cfg.time_start)
    request_end = _to_datetime64(source_cfg.time_end)
    record_start = _to_datetime64(record.time_start)
    record_end = _to_datetime64(record.time_end)
    return (
        (request_start is None or record_start is None or record_start <= request_start)
        and (request_end is None or record_end is None or record_end >= request_end)
    )


def _records_time_bound_request(records: list[_TileRecord], source_cfg: specs.DataSourceConfig) -> bool:
    if not records:
        return False
    request_start = _to_datetime64(source_cfg.time_start)
    request_end = _to_datetime64(source_cfg.time_end)
    starts = [_to_datetime64(record.time_start) for record in records]
    ends = [_to_datetime64(record.time_end) for record in records]
    finite_starts = [value for value in starts if value is not None]
    finite_ends = [value for value in ends if value is not None]
    return (
        request_start is None
        or not finite_starts
        or min(finite_starts) <= request_start
    ) and (
        request_end is None
        or not finite_ends
        or max(finite_ends) >= request_end
    )


def _record_vertical_span(record: _TileRecord) -> float:
    return abs(record.level_max - record.level_min)


def _record_horizontal_area(record: _TileRecord) -> float:
    return abs(record.lat_max - record.lat_min) * abs(record.lon_max - record.lon_min)


def _record_time_span(record: _TileRecord) -> int:
    start = _to_datetime64(record.time_start)
    end = _to_datetime64(record.time_end)
    if start is None or end is None:
        return 0
    return int((end - start) / np.timedelta64(1, "ns"))


def _benchmark_sort_penalty(record: _TileRecord, include_benchmark_variables: bool) -> int:
    if include_benchmark_variables:
        return 0
    return 1 if record.include_benchmark else 0


def _created_at_timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def _optional_float_equal(left, right) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return float(left) == float(right)


def _to_datetime64(value: str | None) -> np.datetime64 | None:
    if value is None:
        return None
    return np.datetime64(value)


def _datetime_sort_value(value: np.datetime64 | None, *, none_first: bool) -> int:
    if value is None:
        return -1 if none_first else 2**63 - 1
    return int(value.astype("datetime64[ns]").astype(np.int64))


def _combine_time_tiles(tiles: list[xr.Dataset]) -> xr.Dataset | None:
    if not tiles:
        return None
    combined = tiles[0]
    for tile in tiles[1:]:
        _validate_non_time_coords_match(combined, tile)
        _validate_time_overlap_compatible(combined, tile)
        tile = _drop_existing_times(tile, combined["time"].values)
        if tile.sizes.get("time", 0) == 0:
            continue
        combined = xr.concat([combined, tile], dim="time").sortby("time")
    return combined


def _drop_existing_times(tile: xr.Dataset, existing_times: np.ndarray) -> xr.Dataset:
    keep = ~np.isin(tile["time"].values, existing_times)
    return tile.isel(time=keep)


def _validate_non_time_coords_match(left: xr.Dataset, right: xr.Dataset) -> None:
    for coord in left.coords:
        if coord == "time":
            continue
        if coord not in right.coords:
            raise OfflineCoverageError(f"Staged ARCO cache tile is missing coordinate {coord!r}.")
        if left[coord].dims != right[coord].dims or not np.array_equal(left[coord].values, right[coord].values):
            raise OfflineCoverageError(
                "Staged ARCO cache cannot mosaic tiles with different horizontal or vertical coordinates."
            )


def _validate_time_overlap_compatible(left: xr.Dataset, right: xr.Dataset) -> None:
    common_times = np.intersect1d(left["time"].values, right["time"].values)
    if common_times.size == 0:
        return
    left_overlap = left.sel(time=common_times)
    right_overlap = right.sel(time=common_times)
    for name in sorted(set(left_overlap.data_vars) & set(right_overlap.data_vars)):
        left_var = left_overlap[name]
        right_var = right_overlap[name]
        if left_var.dims != right_var.dims:
            raise OfflineCoverageError(f"Staged ARCO cache variable {name!r} has incompatible dimensions.")
        if not np.issubdtype(left_var.dtype, np.number) or not np.issubdtype(right_var.dtype, np.number):
            continue
        if not np.allclose(
            left_var.values,
            right_var.values,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        ):
            raise OfflineCoverageError(
                f"Staged ARCO cache conflict while combining overlapping tile values for {name!r}."
            )


def _register_tile(
    root: Path,
    tile_id: str,
    rel_path: Path,
    metadata: _TileWriteMetadata,
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
                metadata.time_start,
                metadata.time_end,
                metadata.lat_min,
                metadata.lat_max,
                metadata.lon_min,
                metadata.lon_max,
                metadata.level_min,
                metadata.level_max,
                1 if include_benchmark_variables else 0,
                json.dumps(asdict(request), sort_keys=True),
                json.dumps(asdict(source_cfg), sort_keys=True),
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def _tile_write_metadata(tile: xr.Dataset) -> _TileWriteMetadata:
    return _TileWriteMetadata(
        sizes=tuple((str(dim), int(size)) for dim, size in tile.sizes.items()),
        variables=tuple(str(name) for name in tile.data_vars),
        time_start=_coord_min_as_str(tile, "time"),
        time_end=_coord_max_as_str(tile, "time"),
        lat_min=float(tile["lat"].min().values),
        lat_max=float(tile["lat"].max().values),
        lon_min=float(tile["lon"].min().values),
        lon_max=float(tile["lon"].max().values),
        level_min=float(tile["level"].min().values),
        level_max=float(tile["level"].max().values),
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


def _new_tmp_tile_path(root: Path, tile_id: str) -> Path:
    tmp_dir = tempfile.mkdtemp(
        prefix=f".{tile_id}.",
        suffix=".tmp.zarr",
        dir=root / TILES_DIR,
    )
    return Path(tmp_dir)


def _validated_tmp_tile_path(root: Path, tmp_path: str | Path, tile_id: str) -> Path:
    path = Path(tmp_path)
    expected_parent = (root / TILES_DIR).resolve()
    if path.parent.resolve() != expected_parent:
        raise ValueError(f"Temporary staged tile must be inside {expected_parent}: {path}")
    if not path.name.startswith(f".{tile_id}.") or not path.name.endswith(".tmp.zarr"):
        raise ValueError(f"Unexpected temporary staged tile name for {tile_id}: {path.name}")
    return path


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
