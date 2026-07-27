"""Finalize yearly staged-cache shards and publish one combined catalog."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import tempfile
from typing import Any

import numpy as np
import xarray as xr

from src import config
from .campaign import CAMPAIGN_FILE, Campaign
from .shard_artifacts import (
    SHARD_MANIFEST_NAME,
    SHARD_SCHEMA_VERSION,
    SUCCESS_MARKER_NAME,
    TILE_COLUMNS,
    ShardValidationError,
    atomic_write_json,
    build_shard_manifest,
    read_tile_rows,
    safe_relative_path,
    sha256_file,
    sha256_json,
    tile_content_signature,
    utc_now,
    validate_completed_shard,
)


CONSOLIDATION_SCHEMA_VERSION = 1
SHARDS_DIR = "shards"
CONSOLIDATION_FILE = "consolidation.json"


def _log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[consolidate] {timestamp} {message}", flush=True)


def year_shard_root(cache_root: str | Path, year: int) -> Path:
    return Path(cache_root) / SHARDS_DIR / f"year={year:04d}"


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


def _validate_row_campaign(row: dict[str, Any], campaign: Campaign) -> None:
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
    if values[0] != np.datetime64(row["time_start"], "ns"):
        raise ShardValidationError(
            f"Tile {row['tile_id']} time_start does not match its Zarr data."
        )
    if values[-1] != np.datetime64(row["time_end"], "ns"):
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
                    f"Year {year} has overlapping time {key} in tiles "
                    f"{previous} and {tile_id}."
                )
            observed_owner[key] = tile_id

    time_start, time_end = campaign.time_window(year)
    start = np.datetime64(time_start, "ns")
    end = np.datetime64(time_end, "ns")
    hour = np.timedelta64(1, "h")
    expected = np.arange(start, end + hour, hour, dtype="datetime64[ns]")
    observed = np.array(sorted(observed_owner), dtype="datetime64[ns]")
    if np.array_equal(observed, expected):
        return
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


def validate_year_shard_content(
    shard_root: Path,
    campaign: Campaign,
    year: int,
) -> tuple[list[dict[str, Any]], str]:
    expected_start, expected_end = campaign.time_window(year)
    rows, schema_sql = read_tile_rows(shard_root)
    times: list[tuple[str, np.ndarray]] = []
    for row in rows:
        _validate_row_campaign(row, campaign)
        if (
            np.datetime64(row["time_start"], "ns")
            < np.datetime64(expected_start, "ns")
            or np.datetime64(row["time_end"], "ns")
            > np.datetime64(expected_end, "ns")
        ):
            raise ShardValidationError(
                f"Tile {row['tile_id']} extends outside its expected yearly season."
            )
        relative = safe_relative_path(str(row["path"]), field="tile path")
        times.append(
            (
                str(row["tile_id"]),
                _tile_times(shard_root.joinpath(*relative.parts), row),
            )
        )
    _validate_year_coverage(year, campaign, times)
    return rows, schema_sql


def finalize_year_shard(
    cache_root: str | Path,
    year: int,
    *,
    pbs_job_id: str | None = None,
    pbs_array_index: str | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    root = Path(cache_root)
    campaign = Campaign.from_file(root / CAMPAIGN_FILE)
    if not campaign.start_year <= year <= campaign.end_year:
        raise ShardValidationError(f"Year {year} is outside the campaign.")
    shard_root = year_shard_root(root, year)
    success_path = shard_root / SUCCESS_MARKER_NAME
    if success_path.exists():
        success, _ = validate_completed_shard(
            shard_root,
            campaign_sha256=campaign.sha256(),
            year=year,
            verify_hashes=True,
        )
        validate_year_shard_content(shard_root, campaign, year)
        if git_commit is not None and success.get("git_commit") != git_commit:
            raise ShardValidationError(
                f"Completed shard year={year} was produced by git commit "
                f"{success.get('git_commit')!r}, not {git_commit!r}."
            )
        return success

    validate_year_shard_content(shard_root, campaign, year)
    time_start, time_end = campaign.time_window(year)
    manifest = build_shard_manifest(
        shard_root,
        campaign_sha256=campaign.sha256(),
        year=year,
        time_start=time_start,
        time_end=time_end,
    )
    manifest_path = shard_root / SHARD_MANIFEST_NAME
    atomic_write_json(manifest_path, manifest)
    success = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "campaign_sha256": campaign.sha256(),
        "year": year,
        "manifest_sha256": sha256_file(manifest_path),
        "finalized_at": utc_now(),
        "pbs_job_id": pbs_job_id,
        "pbs_array_index": pbs_array_index,
        "git_commit": git_commit,
    }
    atomic_write_json(success_path, success)
    validate_completed_shard(
        shard_root,
        campaign_sha256=campaign.sha256(),
        year=year,
        verify_hashes=True,
    )
    _log(f"finalized year={year} shard with {len(manifest['tile_ids'])} tile(s)")
    return success


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
        prefix=".cache.sqlite.",
        suffix=".tmp",
        dir=cache_root,
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
        temp_path.unlink(missing_ok=True)


def consolidate(cache_root: str | Path) -> dict[str, Any]:
    root = Path(cache_root).resolve()
    campaign = Campaign.from_file(root / CAMPAIGN_FILE)
    if root.name != campaign.campaign_id:
        raise ShardValidationError(
            f"Campaign root name {root.name!r} does not match {campaign.campaign_id!r}."
        )
    shards_root = root / SHARDS_DIR
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
    producer_commit: str | None = None
    for year in sorted(expected_years):
        shard_root = year_shard_root(root, year)
        if not shard_root.is_dir():
            raise ShardValidationError(
                f"Missing shard directory for year={year}: {shard_root}"
            )
        _log(f"validating year={year}")
        success, manifest = validate_completed_shard(
            shard_root,
            campaign_sha256=campaign.sha256(),
            year=year,
            verify_hashes=True,
        )
        shard_commit = success.get("git_commit")
        if not isinstance(shard_commit, str) or not shard_commit:
            raise ShardValidationError(
                f"Completed shard year={year} has no producer git commit."
            )
        if producer_commit is None:
            producer_commit = shard_commit
        elif shard_commit != producer_commit:
            raise ShardValidationError(
                f"Year {year} was produced by git commit {shard_commit}, "
                f"not campaign commit {producer_commit}."
            )
        rows, shard_schema = validate_year_shard_content(shard_root, campaign, year)
        if schema_sql is None:
            schema_sql = shard_schema
        elif shard_schema != schema_sql:
            raise ShardValidationError(f"Year {year} uses a different SQLite schema.")

        unique_tile_ids = []
        for row in rows:
            tile_id = str(row["tile_id"])
            relative = safe_relative_path(str(row["path"]), field="tile path")
            signature = tile_content_signature(manifest, str(row["path"]))
            previous = combined_by_tile_id.get(tile_id)
            if previous is not None:
                _, previous_metadata, previous_signature = previous
                if previous_metadata != _canonical_row(row) or previous_signature != signature:
                    raise ShardValidationError(
                        f"Duplicate tile_id {tile_id} has conflicting metadata or content."
                    )
                continue

            combined_row = dict(row)
            combined_row["path"] = PurePosixPath(
                SHARDS_DIR,
                f"year={year:04d}",
                *relative.parts,
            ).as_posix()
            combined_by_tile_id[tile_id] = (
                combined_row,
                _canonical_row(row),
                signature,
            )
            unique_tile_ids.append(tile_id)

        time_start, time_end = campaign.time_window(year)
        shard_summaries.append(
            {
                "year": year,
                "tile_count": len(unique_tile_ids),
                "time_start": time_start,
                "time_end": time_end,
                "manifest_sha256": sha256_json(manifest),
                "git_commit": shard_commit,
            }
        )

    if schema_sql is None:
        raise ShardValidationError("Campaign contains no shard schemas.")
    combined_rows = [
        row_metadata_and_signature[0]
        for _, row_metadata_and_signature in sorted(combined_by_tile_id.items())
    ]
    _create_combined_database(root, schema_sql=schema_sql, rows=combined_rows)

    first_start, _ = campaign.time_window(campaign.start_year)
    _, last_end = campaign.time_window(campaign.end_year)
    summary = {
        "schema_version": CONSOLIDATION_SCHEMA_VERSION,
        "campaign_id": campaign.campaign_id,
        "campaign_sha256": campaign.sha256(),
        "git_commit": producer_commit,
        "time_start": first_start,
        "time_end": last_end,
        "year_count": campaign.task_count,
        "tile_count": len(combined_rows),
        "shards": shard_summaries,
    }
    atomic_write_json(root / CONSOLIDATION_FILE, summary)
    _log(
        f"wrote combined cache.sqlite with {len(combined_rows)} tiles "
        f"covering {campaign.start_year}-{campaign.end_year}"
    )
    return summary
