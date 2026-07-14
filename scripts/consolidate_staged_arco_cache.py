"""Validate yearly staged ARCO shards and build one combined cache catalog."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import sys
import tempfile
from typing import Any

import numpy as np
import xarray as xr

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from deployment.gcp.campaign import Campaign  # noqa: E402
from deployment.gcp.shard_artifacts import (  # noqa: E402
    TILE_COLUMNS,
    ShardValidationError,
    read_tile_rows,
    safe_relative_path,
    sha256_json,
    tile_content_signature,
    validate_completed_shard,
)
from src import config  # noqa: E402


CONSOLIDATION_SCHEMA_VERSION = 1


def _log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[consolidate] {timestamp} {message}", flush=True)


def _canonical_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in TILE_COLUMNS}


def _parse_row_json(row: dict[str, Any], field: str) -> dict[str, Any]:
    try:
        payload = json.loads(str(row[field]))
    except json.JSONDecodeError as exc:
        raise ShardValidationError(
            f"Tile {row['tile_id']} has invalid {field}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ShardValidationError(f"Tile {row['tile_id']} {field} must be an object.")
    return payload


def _validate_row_campaign(
    row: dict[str, Any],
    campaign: Campaign,
) -> None:
    request = _parse_row_json(row, "request_json")
    if request != campaign.request_mapping():
        raise ShardValidationError(
            f"Tile {row['tile_id']} request does not match campaign domain."
        )
    if bool(row["include_benchmark"]) != campaign.include_benchmark_variables:
        raise ShardValidationError(
            f"Tile {row['tile_id']} benchmark setting does not match campaign."
        )
    source = _parse_row_json(row, "source_json")
    if source.get("kind") != "arco_era5":
        raise ShardValidationError(f"Tile {row['tile_id']} is not sourced from ARCO ERA5.")
    if source.get("arco_path") != config.DEFAULT_ARCO_PATH:
        raise ShardValidationError(
            f"Tile {row['tile_id']} uses an unexpected ARCO source path."
        )
    if row["time_start"] is None or row["time_end"] is None:
        raise ShardValidationError(f"Tile {row['tile_id']} is missing time bounds.")
    if (
        np.datetime64(source.get("time_start"), "ns")
        != np.datetime64(row["time_start"], "ns")
        or np.datetime64(source.get("time_end"), "ns")
        != np.datetime64(row["time_end"], "ns")
    ):
        raise ShardValidationError(
            f"Tile {row['tile_id']} source times disagree with its catalog row."
        )


def _tile_times(tile_path: Path, row: dict[str, Any]) -> np.ndarray:
    try:
        with xr.open_zarr(str(tile_path), chunks=None, decode_timedelta=False) as dataset:
            if "time" not in dataset.coords:
                raise ShardValidationError(
                    f"Tile {row['tile_id']} has no time coordinate: {tile_path}"
                )
            values = np.asarray(dataset["time"].values).astype("datetime64[ns]")
    except ShardValidationError:
        raise
    except Exception as exc:
        raise ShardValidationError(
            f"Cannot open Zarr metadata for tile {row['tile_id']}: {exc}"
        ) from exc
    if values.size == 0:
        raise ShardValidationError(f"Tile {row['tile_id']} contains no times.")
    values = np.sort(values)
    if np.unique(values).size != values.size:
        raise ShardValidationError(f"Tile {row['tile_id']} contains duplicate times.")
    if str(values[0]) != str(np.datetime64(row["time_start"], "ns")):
        raise ShardValidationError(
            f"Tile {row['tile_id']} time_start does not match its Zarr data."
        )
    if str(values[-1]) != str(np.datetime64(row["time_end"], "ns")):
        raise ShardValidationError(
            f"Tile {row['tile_id']} time_end does not match its Zarr data."
        )
    return values


def _validate_year_coverage(
    year: int,
    campaign: Campaign,
    tile_times: list[tuple[str, np.ndarray]],
) -> None:
    observed_owner: dict[np.datetime64, str] = {}
    for tile_id, values in tile_times:
        for value in values:
            key = np.datetime64(value, "ns")
            previous = observed_owner.get(key)
            if previous is not None and previous != tile_id:
                raise ShardValidationError(
                    f"Year {year} has overlapping time {key} in tiles {previous} and {tile_id}."
                )
            observed_owner[key] = tile_id

    time_start, time_end = campaign.time_window(year)
    start = np.datetime64(time_start, "ns")
    end = np.datetime64(time_end, "ns")
    hour = np.timedelta64(1, "h")
    expected = np.arange(start, end + hour, hour, dtype="datetime64[ns]")
    observed = np.array(sorted(observed_owner), dtype="datetime64[ns]")
    if not np.array_equal(observed, expected):
        missing = np.setdiff1d(expected, observed, assume_unique=True)
        unexpected = np.setdiff1d(observed, expected, assume_unique=True)
        details = []
        if missing.size:
            details.append(f"missing {missing.size} hour(s), first={missing[0]}")
        if unexpected.size:
            details.append(f"unexpected {unexpected.size} hour(s), first={unexpected[0]}")
        raise ShardValidationError(
            f"Year {year} does not have complete hourly coverage: {'; '.join(details)}"
        )


def _unexpected_year_directories(shards_root: Path, expected: set[int]) -> list[str]:
    unexpected = []
    for path in sorted(shards_root.glob("year=*")):
        suffix = path.name.removeprefix("year=")
        if not path.is_dir() or not suffix.isdigit() or int(suffix) not in expected:
            unexpected.append(path.name)
    return unexpected


def _create_combined_database(
    cache_root: Path,
    *,
    schema_sql: str,
    rows: list[dict[str, Any]],
) -> None:
    descriptor, temp_name = tempfile.mkstemp(
        prefix=".cache.sqlite.", suffix=".tmp", dir=cache_root
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        with sqlite3.connect(temp_path) as conn:
            conn.execute(schema_sql)
            placeholders = ", ".join("?" for _ in TILE_COLUMNS)
            conn.executemany(
                f"INSERT INTO tiles ({', '.join(TILE_COLUMNS)}) VALUES ({placeholders})",
                [tuple(row[column] for column in TILE_COLUMNS) for row in rows],
            )
            integrity = conn.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or integrity[0] != "ok":
                raise ShardValidationError(
                    f"Combined SQLite integrity check failed: {integrity!r}"
                )
        os.replace(temp_path, cache_root / "cache.sqlite")
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def consolidate(cache_root: Path) -> dict[str, Any]:
    cache_root = cache_root.resolve()
    campaign_path = cache_root / "campaign.json"
    campaign = Campaign.from_file(campaign_path)
    campaign_sha256 = campaign.sha256()
    shards_root = cache_root / "shards"
    if not shards_root.is_dir():
        raise ShardValidationError(f"Campaign has no shards directory: {shards_root}")

    expected_years = set(range(campaign.start_year, campaign.end_year + 1))
    unexpected = _unexpected_year_directories(shards_root, expected_years)
    if unexpected:
        raise ShardValidationError(
            f"Campaign contains unexpected shard directories: {', '.join(unexpected)}"
        )

    schema_sql: str | None = None
    combined_by_tile_id: dict[
        str,
        tuple[dict[str, Any], dict[str, Any], str],
    ] = {}
    shard_summaries = []
    for year in sorted(expected_years):
        shard_root = shards_root / f"year={year:04d}"
        if not shard_root.is_dir():
            raise ShardValidationError(f"Missing shard directory for year={year}: {shard_root}")
        _log(f"validating year={year}")
        _, manifest = validate_completed_shard(
            shard_root,
            campaign_sha256=campaign_sha256,
            year=year,
            verify_hashes=True,
        )
        rows, shard_schema = read_tile_rows(shard_root)
        if schema_sql is None:
            schema_sql = shard_schema
        elif shard_schema != schema_sql:
            raise ShardValidationError(f"Year {year} uses a different SQLite schema.")

        expected_start, expected_end = campaign.time_window(year)
        year_times: list[tuple[str, np.ndarray]] = []
        unique_tile_ids = []
        for row in rows:
            _validate_row_campaign(row, campaign)
            tile_id = str(row["tile_id"])
            relative = safe_relative_path(str(row["path"]), field="tile path")
            tile_path = shard_root.joinpath(*relative.parts)
            signature = tile_content_signature(manifest, str(row["path"]))
            previous = combined_by_tile_id.get(tile_id)
            if previous is not None:
                _, previous_metadata, previous_signature = previous
                if previous_metadata != _canonical_row(row) or previous_signature != signature:
                    raise ShardValidationError(
                        f"Duplicate tile_id {tile_id} has conflicting metadata or content."
                    )
                continue

            if (
                np.datetime64(row["time_start"], "ns")
                < np.datetime64(expected_start, "ns")
                or np.datetime64(row["time_end"], "ns")
                > np.datetime64(expected_end, "ns")
            ):
                raise ShardValidationError(
                    f"Tile {tile_id} extends outside its expected yearly season."
                )
            times = _tile_times(tile_path, row)

            combined_row = dict(row)
            nested = PurePosixPath("shards", f"year={year:04d}", *relative.parts)
            combined_row["path"] = nested.as_posix()
            combined_by_tile_id[tile_id] = (
                combined_row,
                _canonical_row(row),
                signature,
            )
            unique_tile_ids.append(tile_id)
            year_times.append((tile_id, times))

        _validate_year_coverage(year, campaign, year_times)
        shard_summaries.append(
            {
                "year": year,
                "tile_count": len(unique_tile_ids),
                "time_start": expected_start,
                "time_end": expected_end,
                "manifest_sha256": sha256_json(manifest),
            }
        )

    assert schema_sql is not None
    combined_rows = [
        row_metadata_and_signature[0]
        for _, row_metadata_and_signature in sorted(combined_by_tile_id.items())
    ]
    _create_combined_database(cache_root, schema_sql=schema_sql, rows=combined_rows)

    first_start, _ = campaign.time_window(campaign.start_year)
    _, last_end = campaign.time_window(campaign.end_year)
    summary = {
        "schema_version": CONSOLIDATION_SCHEMA_VERSION,
        "campaign_id": campaign.campaign_id,
        "campaign_sha256": campaign_sha256,
        "time_start": first_start,
        "time_end": last_end,
        "year_count": campaign.task_count,
        "tile_count": len(combined_rows),
        "shards": shard_summaries,
    }
    _atomic_write_json(cache_root / "consolidation.json", summary)
    _log(
        f"wrote combined cache.sqlite with {len(combined_rows)} tiles "
        f"covering {campaign.start_year}-{campaign.end_year}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        consolidate(args.cache_root)
    except (ShardValidationError, ValueError) as exc:
        print(f"[consolidate:error] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
