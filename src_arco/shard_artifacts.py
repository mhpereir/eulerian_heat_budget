"""Integrity manifests and SQLite validation for staged cache shards."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import tempfile
from typing import Any, Iterable


SHARD_MANIFEST_NAME = "shard-manifest.json"
SUCCESS_MARKER_NAME = "_SUCCESS.json"
SHARD_SCHEMA_VERSION = 1
TILE_COLUMNS = (
    "tile_id",
    "path",
    "time_start",
    "time_end",
    "lat_min",
    "lat_max",
    "lon_min",
    "lon_max",
    "level_min",
    "level_max",
    "include_benchmark",
    "request_json",
    "source_json",
    "created_at",
)


class ShardValidationError(RuntimeError):
    """Raised when a staged cache shard is incomplete or unsafe."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def safe_relative_path(value: str, *, field: str = "path") -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ShardValidationError(f"Unsafe relative {field}: {value!r}")
    return path


def read_tile_rows(cache_root: Path) -> tuple[list[dict[str, Any]], str]:
    database = cache_root / "cache.sqlite"
    if not database.is_file():
        raise ShardValidationError(f"Shard is missing cache.sqlite: {cache_root}")

    uri = f"file:{database.resolve()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or integrity[0] != "ok":
                detail = integrity[0] if integrity else "no result"
                raise ShardValidationError(
                    f"SQLite integrity check failed for {database}: {detail}"
                )
            columns = tuple(
                row[1] for row in conn.execute("PRAGMA table_info(tiles)").fetchall()
            )
            if columns != TILE_COLUMNS:
                raise ShardValidationError(
                    f"Unexpected tiles schema in {database}: {columns!r}"
                )
            schema_row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'tiles'"
            ).fetchone()
            if schema_row is None or not schema_row[0]:
                raise ShardValidationError(f"Missing tiles table in {database}")
            rows = [
                dict(zip(TILE_COLUMNS, row))
                for row in conn.execute(
                    f"SELECT {', '.join(TILE_COLUMNS)} FROM tiles ORDER BY tile_id"
                ).fetchall()
            ]
    except sqlite3.Error as exc:
        raise ShardValidationError(f"Cannot read shard database {database}: {exc}") from exc

    if not rows:
        raise ShardValidationError(f"Shard database contains no tiles: {database}")
    for row in rows:
        relative = safe_relative_path(str(row["path"]), field="tile path")
        tile_path = cache_root.joinpath(*relative.parts)
        if not tile_path.is_dir():
            raise ShardValidationError(
                f"Tile {row['tile_id']} is missing from shard: {relative}"
            )
        if not (tile_path / "zarr.json").is_file() and not (tile_path / ".zgroup").is_file():
            raise ShardValidationError(
                f"Tile {row['tile_id']} has no Zarr group metadata: {relative}"
            )
    return rows, str(schema_row[0])


def _iter_cache_files(cache_root: Path) -> Iterable[Path]:
    excluded = {SHARD_MANIFEST_NAME, SUCCESS_MARKER_NAME}
    for path in sorted(cache_root.rglob("*")):
        relative = path.relative_to(cache_root)
        if path.is_symlink():
            raise ShardValidationError(f"Shard cannot contain symbolic links: {path}")
        if ".write.lock" in relative.parts or any(
            part.startswith(".")
            and (part.endswith(".tmp.zarr") or part.endswith(".tmp"))
            for part in relative.parts
        ):
            raise ShardValidationError(
                f"Shard contains an unfinished cache artifact: {relative}"
            )
        if not path.is_file() or relative.as_posix() in excluded:
            continue
        yield path


def build_shard_manifest(
    cache_root: Path,
    *,
    campaign_sha256: str,
    year: int,
    time_start: str,
    time_end: str,
) -> dict[str, Any]:
    rows, _ = read_tile_rows(cache_root)
    files = [
        {
            "path": path.relative_to(cache_root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _iter_cache_files(cache_root)
    ]
    if not files:
        raise ShardValidationError(f"Shard contains no files: {cache_root}")
    return {
        "schema_version": SHARD_SCHEMA_VERSION,
        "campaign_sha256": campaign_sha256,
        "year": year,
        "time_start": time_start,
        "time_end": time_end,
        "created_at": utc_now(),
        "tile_ids": [str(row["tile_id"]) for row in rows],
        "files": files,
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
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
        temp_path.unlink(missing_ok=True)


def load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ShardValidationError(f"Missing {description}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ShardValidationError(f"Invalid {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ShardValidationError(f"{description} must be a JSON object: {path}")
    return value


def manifest_file_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if manifest.get("schema_version") != SHARD_SCHEMA_VERSION:
        raise ShardValidationError(
            f"Unsupported shard manifest schema: {manifest.get('schema_version')!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ShardValidationError("Shard manifest must contain a non-empty files list.")
    result: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            raise ShardValidationError("Shard manifest contains an invalid file entry.")
        if not isinstance(entry["path"], str):
            raise ShardValidationError("Shard manifest file paths must be strings.")
        path_string = safe_relative_path(
            entry["path"], field="manifest file path"
        ).as_posix()
        if path_string in result:
            raise ShardValidationError(f"Duplicate file in shard manifest: {path_string}")
        size = entry["size"]
        digest = entry["sha256"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ShardValidationError(f"Invalid file size for {path_string}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ShardValidationError(f"Invalid SHA-256 for {path_string}")
        result[path_string] = entry
    return result


def validate_manifest_files(
    shard_root: Path,
    manifest: dict[str, Any],
    *,
    verify_hashes: bool,
) -> None:
    expected = manifest_file_map(manifest)
    observed = {
        path.relative_to(shard_root).as_posix(): path
        for path in _iter_cache_files(shard_root)
    }
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    if missing:
        raise ShardValidationError(f"Shard is missing file(s): {', '.join(missing)}")
    if unexpected:
        raise ShardValidationError(
            f"Shard contains unexpected file(s): {', '.join(unexpected)}"
        )
    for relative, entry in expected.items():
        path = observed[relative]
        if path.stat().st_size != entry["size"]:
            raise ShardValidationError(f"Shard file size mismatch: {relative}")
        if verify_hashes and sha256_file(path) != entry["sha256"]:
            raise ShardValidationError(f"Shard file checksum mismatch: {relative}")


def validate_completed_shard(
    shard_root: Path,
    *,
    campaign_sha256: str,
    year: int,
    verify_hashes: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    success = load_json(
        shard_root / SUCCESS_MARKER_NAME,
        description="shard success marker",
    )
    if success.get("schema_version") != SHARD_SCHEMA_VERSION:
        raise ShardValidationError("Unsupported shard success-marker schema.")
    if success.get("campaign_sha256") != campaign_sha256:
        raise ShardValidationError(
            f"Completed shard year={year} belongs to another campaign configuration."
        )
    if success.get("year") != year:
        raise ShardValidationError(
            f"Completed shard marker year={success.get('year')!r} does not match {year}."
        )

    manifest_path = shard_root / SHARD_MANIFEST_NAME
    manifest = load_json(manifest_path, description="shard manifest")
    if sha256_file(manifest_path) != success.get("manifest_sha256"):
        raise ShardValidationError(f"Shard manifest checksum mismatch for year={year}.")
    if manifest.get("campaign_sha256") != campaign_sha256 or manifest.get("year") != year:
        raise ShardValidationError(f"Shard manifest identity mismatch for year={year}.")
    validate_manifest_files(shard_root, manifest, verify_hashes=verify_hashes)
    read_tile_rows(shard_root)
    return success, manifest


def tile_content_signature(manifest: dict[str, Any], tile_path: str) -> str:
    tile_prefix = (
        safe_relative_path(tile_path, field="tile path").as_posix().rstrip("/") + "/"
    )
    entries = []
    for relative, entry in manifest_file_map(manifest).items():
        if relative.startswith(tile_prefix):
            entries.append(
                {
                    "path": relative[len(tile_prefix) :],
                    "size": entry["size"],
                    "sha256": entry["sha256"],
                }
            )
    if not entries:
        raise ShardValidationError(f"Manifest has no files for tile path {tile_path!r}.")
    return sha256_json(entries)
