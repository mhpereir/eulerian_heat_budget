"""Migrate one year from a legacy indexed cache into a campaign shard."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any

import numpy as np

from .campaign import CAMPAIGN_FILE, Campaign
from .consolidation import (
    finalize_year_shard,
    validate_year_shard_content,
    year_shard_root,
)
from .shard_artifacts import (
    SUCCESS_MARKER_NAME,
    TILE_COLUMNS,
    ShardValidationError,
    read_tile_rows,
    safe_relative_path,
    sha256_file,
    validate_completed_shard,
)


def _require_distinct_roots(source_root: Path, campaign_root: Path) -> None:
    source = source_root.resolve()
    destination = campaign_root.resolve()
    if source == destination:
        raise ShardValidationError(
            "Legacy source and campaign destination must be different directories."
        )
    if source in destination.parents:
        raise ShardValidationError(
            "Campaign destination cannot be nested inside the legacy source."
        )
    if destination in source.parents:
        raise ShardValidationError(
            "Legacy source cannot be nested inside the campaign destination."
        )


def _row_year(row: dict[str, Any]) -> int:
    try:
        start = np.datetime64(row["time_start"], "ns")
        end = np.datetime64(row["time_end"], "ns")
    except (TypeError, ValueError) as exc:
        raise ShardValidationError(
            f"Tile {row['tile_id']} has invalid time bounds."
        ) from exc
    start_year = int(str(start)[:4])
    end_year = int(str(end)[:4])
    if start_year != end_year:
        raise ShardValidationError(
            f"Tile {row['tile_id']} crosses a year boundary."
        )
    return start_year


def _iter_regular_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ShardValidationError(
                f"Legacy migration does not accept symbolic links: {path}"
            )
        if path.is_file():
            files[relative] = path
    return files


def _validate_existing_copy_tree(source: Path, destination: Path) -> None:
    source_files = _iter_regular_files(source)
    destination_files = _iter_regular_files(destination)
    if set(source_files) != set(destination_files):
        raise ShardValidationError(
            f"Existing destination tile does not match the legacy source: {destination}"
        )
    for relative, source_file in source_files.items():
        destination_file = destination_files[relative]
        source_stat = source_file.stat()
        destination_stat = destination_file.stat()
        if source_stat.st_size != destination_stat.st_size:
            raise ShardValidationError(
                "Existing destination file size differs from the legacy source: "
                f"{destination_file}"
            )
        if os.path.samestat(source_stat, destination_stat):
            raise ShardValidationError(
                f"Destination file must be an independent copy: {destination_file}"
            )
        if sha256_file(source_file) != sha256_file(destination_file):
            raise ShardValidationError(
                "Existing destination file content differs from the legacy source: "
                f"{destination_file}"
            )


def _copy_tile(source: Path, destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise ShardValidationError(
                f"Destination tile path is not a directory: {destination}"
            )
        _validate_existing_copy_tree(source, destination)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.legacy-migration.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    temp_dir.rmdir()
    try:
        shutil.copytree(source, temp_dir, copy_function=shutil.copy2)
        os.replace(temp_dir, destination)
        _validate_existing_copy_tree(source, destination)
    except FileExistsError:
        _validate_existing_copy_tree(source, destination)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def _create_shard_database(
    database: Path,
    *,
    schema_sql: str,
    rows: list[dict[str, Any]],
) -> None:
    if database.exists():
        existing_rows, existing_schema = read_tile_rows(database.parent)
        if existing_schema != schema_sql or existing_rows != rows:
            raise ShardValidationError(
                f"Existing shard database conflicts with migration input: {database}"
            )
        return

    descriptor, temp_name = tempfile.mkstemp(
        prefix=".cache.sqlite.",
        suffix=".tmp",
        dir=database.parent,
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        with sqlite3.connect(temp_path) as connection:
            connection.execute(schema_sql)
            placeholders = ", ".join("?" for _ in TILE_COLUMNS)
            connection.executemany(
                f"INSERT INTO tiles ({', '.join(TILE_COLUMNS)}) VALUES ({placeholders})",
                [tuple(row[column] for column in TILE_COLUMNS) for row in rows],
            )
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or integrity[0] != "ok":
                raise ShardValidationError(
                    f"Migrated SQLite integrity check failed: {integrity!r}"
                )
        os.replace(temp_path, database)
    finally:
        temp_path.unlink(missing_ok=True)


def migrate_legacy_year(
    legacy_cache_root: str | Path,
    campaign_root: str | Path,
    year: int,
    *,
    pbs_job_id: str | None = None,
    pbs_array_index: str | None = None,
    git_commit: str,
) -> dict[str, Any]:
    """Copy one legacy year, validate it, and publish shard provenance."""
    source_root = Path(legacy_cache_root)
    destination_root = Path(campaign_root)
    _require_distinct_roots(source_root, destination_root)
    campaign = Campaign.from_file(destination_root / CAMPAIGN_FILE)
    if not campaign.start_year <= year <= campaign.end_year:
        raise ShardValidationError(f"Year {year} is outside the campaign.")

    shard_root = year_shard_root(destination_root, year)
    success_path = shard_root / SUCCESS_MARKER_NAME
    if success_path.exists():
        success, _ = validate_completed_shard(
            shard_root,
            campaign_sha256=campaign.sha256(),
            year=year,
            verify_hashes=True,
        )
        validate_year_shard_content(shard_root, campaign, year)
        if success.get("git_commit") != git_commit:
            raise ShardValidationError(
                f"Completed shard year={year} was produced by Git commit "
                f"{success.get('git_commit')!r}, not {git_commit!r}."
            )
        return success

    source_rows, schema_sql = read_tile_rows(source_root)
    rows = [row for row in source_rows if _row_year(row) == year]
    if not rows:
        raise ShardValidationError(
            f"Legacy cache has no tiles for campaign year={year}."
        )
    rows.sort(key=lambda row: str(row["tile_id"]))

    tiles_root = shard_root / "tiles"
    tiles_root.mkdir(parents=True, exist_ok=True)
    expected_tile_names: set[str] = set()
    for row in rows:
        relative = safe_relative_path(str(row["path"]), field="legacy tile path")
        if len(relative.parts) != 2 or relative.parts[0] != "tiles":
            raise ShardValidationError(
                f"Legacy tile path does not use tiles/<tile-id>.zarr: {relative}"
            )
        expected_tile_names.add(relative.parts[1])
        source_tile = source_root.joinpath(*relative.parts)
        destination_tile = shard_root.joinpath(*relative.parts)
        _copy_tile(source_tile, destination_tile)

    unexpected = sorted(
        path.name
        for path in tiles_root.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and path.name not in expected_tile_names
    )
    if unexpected:
        raise ShardValidationError(
            f"Year {year} shard contains unexpected tile directories: "
            f"{', '.join(unexpected)}"
        )

    _create_shard_database(
        shard_root / "cache.sqlite",
        schema_sql=schema_sql,
        rows=rows,
    )
    return finalize_year_shard(
        destination_root,
        year,
        pbs_job_id=pbs_job_id,
        pbs_array_index=pbs_array_index,
        git_commit=git_commit,
    )
