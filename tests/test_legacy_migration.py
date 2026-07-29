from pathlib import Path
import sqlite3

import pytest

from src_arco.campaign import Campaign, initialize_campaign
from src_arco.legacy_migration import migrate_legacy_year
from src_arco.shard_artifacts import TILE_COLUMNS, ShardValidationError


SCHEMA_SQL = """CREATE TABLE tiles (
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
)"""


def _campaign() -> Campaign:
    return Campaign.from_mapping(
        {
            "schema_version": 1,
            "campaign_id": "pnw-bartusek-surface-700hpa-1940-1940",
            "years": {"start": 1940, "end": 1940},
            "season": {
                "start_month_day": "05-01",
                "end_month_day": "05-01",
            },
            "domain": {
                "region": "pnw_bartusek",
                "margin_n": 1,
                "zg_top_pa": 70000,
                "zg_bottom": "surface_pressure",
                "allow_bottom_overflow": True,
            },
            "staging": {
                "time_chunk": "month",
                "attempt_timeout_seconds": 20800,
                "include_benchmark_variables": False,
            },
        }
    )


def _legacy_cache(root: Path) -> Path:
    cache_root = root / "legacy"
    tile_root = cache_root / "tiles" / "tile-1940.zarr"
    tile_root.mkdir(parents=True)
    (tile_root / "zarr.json").write_text("{}\n", encoding="utf-8")
    (tile_root / "chunk").write_bytes(b"legacy-data")
    request = (
        '{"bbox": [40.0, 60.0, -130.0, -110.0], "margin_n": 1, '
        '"zg_bottom": "surface_pressure", "zg_bottom_pressure": null, '
        '"zg_top_pressure": 70000.0}'
    )
    source = (
        '{"kind": "arco_era5", '
        '"arco_path": "gs://gcp-public-data-arco-era5/ar/'
        'full_37-1h-0p25deg-chunk-1.zarr-v3", '
        '"time_start": "1940-05-01T00:00:00", '
        '"time_end": "1940-05-01T23:00:00"}'
    )
    row = (
        "tile-1940",
        "tiles/tile-1940.zarr",
        "1940-05-01T00:00:00.000000000",
        "1940-05-01T23:00:00.000000000",
        40.0,
        60.0,
        -130.0,
        -110.0,
        70000.0,
        100000.0,
        0,
        request,
        source,
        "2026-07-01T00:00:00+00:00",
    )
    with sqlite3.connect(cache_root / "cache.sqlite") as connection:
        connection.execute(SCHEMA_SQL)
        placeholders = ", ".join("?" for _ in TILE_COLUMNS)
        connection.execute(
            f"INSERT INTO tiles ({', '.join(TILE_COLUMNS)}) VALUES ({placeholders})",
            row,
        )
    return cache_root


def test_migrate_legacy_year_copies_and_finalizes(monkeypatch, tmp_path):
    source = _legacy_cache(tmp_path)
    campaign_root = tmp_path / _campaign().campaign_id
    initialize_campaign(campaign_root, _campaign())

    monkeypatch.setattr(
        "src_arco.legacy_migration.finalize_year_shard",
        lambda *args, **kwargs: {
            "year": 1940,
            "manifest_sha256": "f" * 64,
        },
    )
    result = migrate_legacy_year(
        source,
        campaign_root,
        1940,
        git_commit="a" * 40,
    )

    source_chunk = source / "tiles" / "tile-1940.zarr" / "chunk"
    migrated_chunk = (
        campaign_root
        / "shards"
        / "year=1940"
        / "tiles"
        / "tile-1940.zarr"
        / "chunk"
    )
    assert result["year"] == 1940
    assert source_chunk.read_bytes() == migrated_chunk.read_bytes()
    assert (
        source_chunk.stat().st_dev,
        source_chunk.stat().st_ino,
    ) != (
        migrated_chunk.stat().st_dev,
        migrated_chunk.stat().st_ino,
    )
    with sqlite3.connect(
        campaign_root / "shards" / "year=1940" / "cache.sqlite"
    ) as connection:
        assert connection.execute("SELECT COUNT(*) FROM tiles").fetchone() == (1,)


def test_migrate_legacy_year_is_restartable_before_finalization(monkeypatch, tmp_path):
    source = _legacy_cache(tmp_path)
    campaign_root = tmp_path / _campaign().campaign_id
    initialize_campaign(campaign_root, _campaign())
    monkeypatch.setattr(
        "src_arco.legacy_migration.finalize_year_shard",
        lambda *args, **kwargs: {
            "year": 1940,
            "manifest_sha256": "f" * 64,
        },
    )

    first = migrate_legacy_year(
        source,
        campaign_root,
        1940,
        git_commit="a" * 40,
    )
    second = migrate_legacy_year(
        source,
        campaign_root,
        1940,
        git_commit="a" * 40,
    )

    assert first == second


def test_migrate_legacy_year_rejects_conflicting_destination(monkeypatch, tmp_path):
    source = _legacy_cache(tmp_path)
    campaign_root = tmp_path / _campaign().campaign_id
    initialize_campaign(campaign_root, _campaign())
    destination_tile = (
        campaign_root
        / "shards"
        / "year=1940"
        / "tiles"
        / "tile-1940.zarr"
    )
    destination_tile.mkdir(parents=True)
    (destination_tile / "zarr.json").write_text("{}\n", encoding="utf-8")
    (destination_tile / "chunk").write_bytes(b"different")
    monkeypatch.setattr(
        "src_arco.legacy_migration.finalize_year_shard",
        lambda *args, **kwargs: {},
    )

    with pytest.raises(ShardValidationError, match="differs from the legacy source"):
        migrate_legacy_year(
            source,
            campaign_root,
            1940,
            git_commit="a" * 40,
        )
