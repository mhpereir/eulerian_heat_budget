import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

import numpy as np
import pytest
import xarray as xr

from deployment.gcp.campaign import Campaign, CampaignConfigError
from deployment.gcp.render_batch_job import render_job
from deployment.gcp.run_yearly_retrieval import (
    build_retrieval_command,
    run_from_environment,
)
from deployment.gcp.shard_artifacts import (
    SHARD_MANIFEST_NAME,
    SUCCESS_MARKER_NAME,
    ShardValidationError,
    build_shard_manifest,
    sha256_file,
    write_json,
    write_shard_manifest,
)
from scripts.consolidate_staged_arco_cache import consolidate
from scripts import consolidate_staged_arco_cache as consolidation_module
from src import config


TILES_SCHEMA = """
CREATE TABLE tiles (
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


@pytest.fixture(autouse=True)
def _open_test_zarr(monkeypatch):
    def fake_open_zarr(path, *args, **kwargs):
        tile_path = Path(path)
        times = np.array(
            json.loads((tile_path / "time-values.json").read_text(encoding="utf-8")),
            dtype="datetime64[ns]",
        )
        return xr.Dataset(coords={"time": times})

    monkeypatch.setattr(consolidation_module.xr, "open_zarr", fake_open_zarr)


def _campaign_mapping(*, start_year=2000, end_year=2000):
    return {
        "campaign_id": "test-campaign",
        "years": {"start": start_year, "end": end_year},
        "season": {"start_month_day": "01-01", "end_month_day": "01-01"},
        "domain": {
            "region": "eastern_canada",
            "margin_n": 1,
            "zg_top_pa": 70000,
            "zg_bottom": "surface_pressure",
            "allow_bottom_overflow": True,
        },
        "staging": {
            "time_chunk": "day",
            "attempt_timeout_seconds": 300,
            "include_benchmark_variables": False,
        },
    }


def _campaign(*, start_year=2000, end_year=2000):
    return Campaign.from_mapping(
        _campaign_mapping(start_year=start_year, end_year=end_year)
    )


def _write_cache_tile(cache_root: Path, campaign: Campaign, year: int, *, gap=False):
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "tiles").mkdir(exist_ok=True)
    time_start, time_end = campaign.time_window(year)
    times = np.arange(
        np.datetime64(time_start, "ns"),
        np.datetime64(time_end, "ns") + np.timedelta64(1, "h"),
        np.timedelta64(1, "h"),
    )
    if gap:
        times = np.delete(times, len(times) // 2)
    tile_id = f"tile-{year}"
    relative = Path("tiles") / f"{tile_id}.zarr"
    tile_path = cache_root / relative
    tile_path.mkdir(parents=True)
    (tile_path / "zarr.json").write_text("{}\n", encoding="utf-8")
    (tile_path / "time-values.json").write_text(
        json.dumps([str(value) for value in times]) + "\n",
        encoding="utf-8",
    )

    row_start = str(times[0])
    row_end = str(times[-1])
    source = {
        "kind": "arco_era5",
        "path_data": None,
        "arco_path": config.DEFAULT_ARCO_PATH,
        "arco_storage_token": config.DEFAULT_ARCO_TOKEN,
        "chunks_time": config.n_time,
        "staged_cache_root": None,
        "time_start": row_start,
        "time_end": row_end,
    }
    with sqlite3.connect(cache_root / "cache.sqlite") as conn:
        conn.execute(TILES_SCHEMA)
        conn.execute(
            """
            INSERT INTO tiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tile_id,
                relative.as_posix(),
                row_start,
                row_end,
                42.0,
                52.0,
                -83.0,
                -73.0,
                70000.0,
                100000.0,
                0,
                json.dumps(campaign.request_mapping(), sort_keys=True),
                json.dumps(source, sort_keys=True),
                "2000-01-01T00:00:00+00:00",
            ),
        )
    return tile_id


def _finalize_shard(shard_root: Path, campaign: Campaign, year: int):
    time_start, time_end = campaign.time_window(year)
    manifest = build_shard_manifest(
        shard_root,
        campaign_sha256=campaign.sha256(),
        year=year,
        time_start=time_start,
        time_end=time_end,
    )
    manifest_path = write_shard_manifest(shard_root, manifest)
    write_json(
        shard_root / SUCCESS_MARKER_NAME,
        {
            "schema_version": 1,
            "campaign_sha256": campaign.sha256(),
            "year": year,
            "manifest_sha256": sha256_file(manifest_path),
        },
    )


def _write_campaign_tree(root: Path, campaign: Campaign, *, gap_year=None):
    campaign.write_normalized(root / "campaign.json")
    for year in range(campaign.start_year, campaign.end_year + 1):
        shard = root / "shards" / f"year={year:04d}"
        _write_cache_tile(shard, campaign, year, gap=year == gap_year)
        _finalize_shard(shard, campaign, year)


def _refresh_manifest_file(shard: Path, relative: str):
    manifest_path = shard / SHARD_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = shard / relative
    for entry in manifest["files"]:
        if entry["path"] == relative:
            entry["size"] = target.stat().st_size
            entry["sha256"] = sha256_file(target)
            break
    else:
        raise AssertionError(f"missing manifest entry for {relative}")
    write_json(manifest_path, manifest)
    success_path = shard / SUCCESS_MARKER_NAME
    success = json.loads(success_path.read_text(encoding="utf-8"))
    success["manifest_sha256"] = sha256_file(manifest_path)
    write_json(success_path, success)


def test_campaign_is_normalized_hashed_and_maps_tasks(tmp_path):
    campaign = _campaign(start_year=2000, end_year=2002)
    normalized = tmp_path / "campaign.json"
    campaign.write_normalized(normalized)

    reloaded = Campaign.from_file(normalized)

    assert reloaded == campaign
    assert reloaded.sha256() == campaign.sha256()
    assert campaign.task_count == 3
    assert campaign.year_for_task(2, 3) == 2002
    assert campaign.time_window(2000) == (
        "2000-01-01T00:00:00",
        "2000-01-01T23:00:00",
    )


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda payload: payload["domain"].update(
                {
                    "bbox": {
                        "lat_min": 42,
                        "lat_max": 52,
                        "lon_min": -83,
                        "lon_max": -73,
                    }
                }
            ),
            "exactly one",
        ),
        (
            lambda payload: payload["season"].update(
                {"start_month_day": "11-01", "end_month_day": "02-01"}
            ),
            "cannot cross",
        ),
        (
            lambda payload: payload["domain"].update(
                {"zg_bottom": "pressure_level"}
            ),
            "zg_bottom_pa is required",
        ),
    ],
)
def test_campaign_rejects_ambiguous_or_incompatible_inputs(mutate, match):
    payload = _campaign_mapping()
    mutate(payload)

    with pytest.raises(CampaignConfigError, match=match):
        Campaign.from_mapping(payload)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda payload: payload["staging"].pop("time_chunk"), "missing required"),
        (
            lambda payload: payload["season"].update(
                {"start_month_day": "02-29", "end_month_day": "02-29"}
            ),
            "invalid date for year 2001",
        ),
        (
            lambda payload: payload["domain"].update({"region": "not-a-region"}),
            "must be one of",
        ),
        (
            lambda payload: payload["domain"].update({"zg_bottom_pa": 90000}),
            "must be omitted",
        ),
    ],
)
def test_campaign_rejects_missing_invalid_and_incompatible_values(mutate, match):
    payload = _campaign_mapping(start_year=2001, end_year=2001)
    mutate(payload)

    with pytest.raises(CampaignConfigError, match=match):
        Campaign.from_mapping(payload)


def test_render_job_uses_one_digest_pinned_task_per_year():
    campaign = _campaign(start_year=2000, end_year=2002)
    digest = "sha256:" + "a" * 64

    job = render_job(
        campaign,
        project="example-project",
        region="us-central1",
        bucket="example-bucket",
        image_uri=f"us-central1-docker.pkg.dev/example/repo/image@{digest}",
        service_account_email="batch@example-project.iam.gserviceaccount.com",
        job_id="test-campaign-job",
        parallelism=2,
        boot_disk_gb=100,
    )

    group = job["taskGroups"][0]
    assert group["taskCount"] == 3
    assert group["parallelism"] == 2
    assert group["taskCountPerNode"] == 1
    assert group["taskSpec"]["computeResource"] == {
        "cpuMilli": 2000,
        "memoryMib": 7000,
    }
    assert group["taskSpec"]["environment"]["variables"]["DASK_NUM_WORKERS"] == "2"
    assert group["taskSpec"]["volumes"][0]["mountOptions"] == [
        "--implicit-dirs",
        "--uid=10001",
        "--gid=10001",
        "--file-mode=0660",
        "--dir-mode=0770",
    ]
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "e2-standard-2"
    assert job["allocationPolicy"]["instances"][0]["policy"]["bootDisk"]["sizeGb"] == 100

    with pytest.raises(ValueError, match="digest"):
        render_job(
            campaign,
            project="example-project",
            region="us-central1",
            bucket="example-bucket",
            image_uri="us-central1-docker.pkg.dev/example/repo/image:latest",
            service_account_email="batch@example-project.iam.gserviceaccount.com",
            job_id="test-campaign-job",
        )


def test_retrieval_command_contains_explicit_campaign_parameters(tmp_path):
    campaign = _campaign()

    command = build_retrieval_command(
        campaign,
        year=2000,
        cache_root=tmp_path,
        python_executable="python-test",
    )

    assert command[0] == "python-test"
    assert command[command.index("--region") + 1] == "eastern_canada"
    assert command[command.index("--time-start") + 1] == "2000-01-01T00:00:00"
    assert command[command.index("--time-end") + 1] == "2000-01-01T23:00:00"
    assert "--no-use-surface-variables" in command
    assert "--allow-bottom-overflow" in command


def test_retrieval_command_supports_explicit_bbox_and_pressure_bottom(tmp_path):
    payload = _campaign_mapping()
    payload["domain"].pop("region")
    payload["domain"].update(
        {
            "bbox": {
                "lat_min": 42,
                "lat_max": 52,
                "lon_min": -83,
                "lon_max": -73,
            },
            "zg_bottom": "pressure_level",
            "zg_bottom_pa": 90000,
            "allow_bottom_overflow": False,
        }
    )
    campaign = Campaign.from_mapping(payload)

    command = build_retrieval_command(campaign, year=2000, cache_root=tmp_path)

    assert "--region" not in command
    assert command[command.index("--lat-min") + 1] == "42.0"
    assert command[command.index("--zg-bottom-pa") + 1] == "90000.0"
    assert "--no-allow-bottom-overflow" in command


def test_batch_entrypoint_publishes_then_skips_completed_shard(tmp_path):
    campaign = _campaign()
    scratch = tmp_path / "scratch"
    output = tmp_path / "output"
    calls = []

    def fake_runner(command, **kwargs):
        calls.append(command)
        cache_root = Path(command[command.index("--staged-cache-root") + 1])
        _write_cache_tile(cache_root, campaign, 2000)
        return subprocess.CompletedProcess(command, 0)

    environ = {
        "EHB_CAMPAIGN_JSON": campaign.canonical_json(),
        "EHB_CAMPAIGN_SHA256": campaign.sha256(),
        "EHB_LOCAL_SCRATCH": str(scratch),
        "EHB_OUTPUT_MOUNT": str(output),
        "BATCH_TASK_INDEX": "0",
        "BATCH_TASK_COUNT": "1",
        "BATCH_TASK_RETRY_ATTEMPT": "0",
    }

    assert run_from_environment(environ, command_runner=fake_runner) == 0
    shard = output / "shards" / "year=2000"
    assert (shard / SUCCESS_MARKER_NAME).is_file()
    assert (shard / SHARD_MANIFEST_NAME).is_file()

    assert run_from_environment(
        environ,
        command_runner=lambda *args, **kwargs: pytest.fail("retrieval should be skipped"),
    ) == 0
    assert len(calls) == 1


def test_batch_entrypoint_replaces_partial_publication(tmp_path):
    campaign = _campaign()
    scratch = tmp_path / "scratch"
    output = tmp_path / "output"
    partial = output / "shards/year=2000/partial.txt"
    partial.parent.mkdir(parents=True)
    partial.write_text("incomplete", encoding="utf-8")

    def fake_runner(command, **kwargs):
        cache_root = Path(command[command.index("--staged-cache-root") + 1])
        _write_cache_tile(cache_root, campaign, 2000)
        return subprocess.CompletedProcess(command, 0)

    environ = {
        "EHB_CAMPAIGN_JSON": campaign.canonical_json(),
        "EHB_CAMPAIGN_SHA256": campaign.sha256(),
        "EHB_LOCAL_SCRATCH": str(scratch),
        "EHB_OUTPUT_MOUNT": str(output),
        "BATCH_TASK_INDEX": "0",
        "BATCH_TASK_COUNT": "1",
    }

    run_from_environment(environ, command_runner=fake_runner)

    assert not partial.exists()
    assert (output / "shards/year=2000" / SUCCESS_MARKER_NAME).is_file()


def test_batch_entrypoint_rejects_completed_shard_from_other_configuration(tmp_path):
    original = _campaign()
    output = tmp_path / "output"
    shard = output / "shards/year=2000"
    _write_cache_tile(shard, original, 2000)
    _finalize_shard(shard, original, 2000)
    changed_payload = _campaign_mapping()
    changed_payload["domain"]["margin_n"] = 2
    changed = Campaign.from_mapping(changed_payload)
    environ = {
        "EHB_CAMPAIGN_JSON": changed.canonical_json(),
        "EHB_CAMPAIGN_SHA256": changed.sha256(),
        "EHB_LOCAL_SCRATCH": str(tmp_path / "scratch"),
        "EHB_OUTPUT_MOUNT": str(output),
        "BATCH_TASK_INDEX": "0",
        "BATCH_TASK_COUNT": "1",
    }

    with pytest.raises(ShardValidationError, match="different campaign"):
        run_from_environment(environ)


def test_batch_entrypoint_rejects_task_count_mismatch(tmp_path):
    campaign = _campaign(start_year=2000, end_year=2001)
    environ = {
        "EHB_CAMPAIGN_JSON": campaign.canonical_json(),
        "EHB_CAMPAIGN_SHA256": campaign.sha256(),
        "EHB_LOCAL_SCRATCH": str(tmp_path / "scratch"),
        "EHB_OUTPUT_MOUNT": str(tmp_path / "output"),
        "BATCH_TASK_INDEX": "0",
        "BATCH_TASK_COUNT": "1",
    }

    with pytest.raises(CampaignConfigError, match="does not match"):
        run_from_environment(environ)


def test_consolidation_builds_nested_catalog_and_is_resumable(tmp_path):
    campaign = _campaign(start_year=2000, end_year=2001)
    _write_campaign_tree(tmp_path, campaign)

    first = consolidate(tmp_path)
    with sqlite3.connect(tmp_path / "cache.sqlite") as conn:
        paths = [row[0] for row in conn.execute("SELECT path FROM tiles ORDER BY path")]
    first_database = (tmp_path / "cache.sqlite").read_bytes()
    second = consolidate(tmp_path)

    assert first["tile_count"] == 2
    assert second["tile_count"] == 2
    assert paths == [
        "shards/year=2000/tiles/tile-2000.zarr",
        "shards/year=2001/tiles/tile-2001.zarr",
    ]
    assert second == first
    assert (tmp_path / "cache.sqlite").read_bytes() == first_database


def test_consolidation_rejects_missing_shard_marker(tmp_path):
    campaign = _campaign()
    _write_campaign_tree(tmp_path, campaign)
    (tmp_path / "shards/year=2000" / SUCCESS_MARKER_NAME).unlink()

    with pytest.raises(ShardValidationError, match="success marker"):
        consolidate(tmp_path)


def test_consolidation_rejects_missing_and_unexpected_years(tmp_path):
    campaign = _campaign(start_year=2000, end_year=2001)
    _write_campaign_tree(tmp_path, campaign)
    shutil.rmtree(tmp_path / "shards/year=2001")

    with pytest.raises(ShardValidationError, match="Missing shard directory"):
        consolidate(tmp_path)

    (tmp_path / "shards/year=2002").mkdir()
    with pytest.raises(ShardValidationError, match="unexpected shard directories"):
        consolidate(tmp_path)


def test_consolidation_rejects_missing_tile_files(tmp_path):
    campaign = _campaign()
    _write_campaign_tree(tmp_path, campaign)
    shutil.rmtree(tmp_path / "shards/year=2000/tiles/tile-2000.zarr")

    with pytest.raises(ShardValidationError, match="missing published file"):
        consolidate(tmp_path)


def test_consolidation_rejects_corrupted_file_and_preserves_catalog(tmp_path):
    campaign = _campaign()
    _write_campaign_tree(tmp_path, campaign)
    consolidate(tmp_path)
    original_database = (tmp_path / "cache.sqlite").read_bytes()
    shard = tmp_path / "shards/year=2000"
    data_file = next(
        path
        for path in (shard / "tiles/tile-2000.zarr").rglob("*")
        if path.is_file() and path.name not in {"zarr.json", ".zgroup"}
    )
    data_file.write_bytes(data_file.read_bytes() + b"corruption")

    with pytest.raises(ShardValidationError, match="size mismatch|checksum mismatch"):
        consolidate(tmp_path)
    assert (tmp_path / "cache.sqlite").read_bytes() == original_database


def test_consolidation_rejects_hourly_gap(tmp_path):
    campaign = _campaign()
    _write_campaign_tree(tmp_path, campaign, gap_year=2000)

    with pytest.raises(ShardValidationError, match="complete hourly coverage"):
        consolidate(tmp_path)


def test_consolidation_rejects_conflicting_time_overlap(tmp_path):
    campaign = _campaign()
    _write_campaign_tree(tmp_path, campaign)
    shard = tmp_path / "shards/year=2000"
    original_path = shard / "tiles/tile-2000.zarr"
    overlap_path = shard / "tiles/overlap.zarr"
    shutil.copytree(original_path, overlap_path)
    with sqlite3.connect(shard / "cache.sqlite") as conn:
        original = list(
            conn.execute("SELECT * FROM tiles WHERE tile_id = 'tile-2000'").fetchone()
        )
        original[0] = "overlap"
        original[1] = "tiles/overlap.zarr"
        conn.execute(
            "INSERT INTO tiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            original,
        )
    (shard / SHARD_MANIFEST_NAME).unlink()
    (shard / SUCCESS_MARKER_NAME).unlink()
    _finalize_shard(shard, campaign, 2000)

    with pytest.raises(ShardValidationError, match="overlapping time"):
        consolidate(tmp_path)


def test_consolidation_rejects_path_traversal(tmp_path):
    campaign = _campaign()
    _write_campaign_tree(tmp_path, campaign)
    shard = tmp_path / "shards/year=2000"
    with sqlite3.connect(shard / "cache.sqlite") as conn:
        conn.execute("UPDATE tiles SET path = '../outside.zarr'")
    _refresh_manifest_file(shard, "cache.sqlite")

    with pytest.raises(ShardValidationError, match="Unsafe relative tile path"):
        consolidate(tmp_path)


def test_consolidation_deduplicates_identical_redundant_tile(tmp_path):
    campaign = _campaign(start_year=2000, end_year=2001)
    _write_campaign_tree(tmp_path, campaign)
    first = tmp_path / "shards/year=2000"
    second = tmp_path / "shards/year=2001"
    shutil.copytree(
        first / "tiles/tile-2000.zarr",
        second / "tiles/tile-2000.zarr",
    )
    with sqlite3.connect(first / "cache.sqlite") as first_db:
        duplicate = first_db.execute("SELECT * FROM tiles WHERE tile_id = 'tile-2000'").fetchone()
    with sqlite3.connect(second / "cache.sqlite") as second_db:
        second_db.execute(
            "INSERT INTO tiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            duplicate,
        )
    (second / SHARD_MANIFEST_NAME).unlink()
    (second / SUCCESS_MARKER_NAME).unlink()
    _finalize_shard(second, campaign, 2001)

    summary = consolidate(tmp_path)

    assert summary["tile_count"] == 2


def test_consolidation_rejects_conflicting_duplicate_tile(tmp_path):
    campaign = _campaign(start_year=2000, end_year=2001)
    _write_campaign_tree(tmp_path, campaign)
    first = tmp_path / "shards/year=2000"
    second = tmp_path / "shards/year=2001"
    duplicate_path = second / "tiles/tile-2000.zarr"
    shutil.copytree(first / "tiles/tile-2000.zarr", duplicate_path)
    data_file = next(path for path in duplicate_path.rglob("*") if path.is_file())
    data_file.write_bytes(data_file.read_bytes() + b"different")
    with sqlite3.connect(first / "cache.sqlite") as first_db:
        duplicate = first_db.execute("SELECT * FROM tiles WHERE tile_id = 'tile-2000'").fetchone()
    with sqlite3.connect(second / "cache.sqlite") as second_db:
        second_db.execute(
            "INSERT INTO tiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            duplicate,
        )
    (second / SHARD_MANIFEST_NAME).unlink()
    (second / SUCCESS_MARKER_NAME).unlink()
    _finalize_shard(second, campaign, 2001)

    with pytest.raises(ShardValidationError, match="conflicting metadata or content"):
        consolidate(tmp_path)


def test_consolidation_rejects_duplicate_with_different_database_metadata(tmp_path):
    campaign = _campaign(start_year=2000, end_year=2001)
    _write_campaign_tree(tmp_path, campaign)
    first = tmp_path / "shards/year=2000"
    second = tmp_path / "shards/year=2001"
    shutil.copytree(
        first / "tiles/tile-2000.zarr",
        second / "tiles/tile-2000.zarr",
    )
    with sqlite3.connect(first / "cache.sqlite") as first_db:
        duplicate = list(
            first_db.execute("SELECT * FROM tiles WHERE tile_id = 'tile-2000'").fetchone()
        )
    duplicate[-1] = "2000-01-02T00:00:00+00:00"
    with sqlite3.connect(second / "cache.sqlite") as second_db:
        second_db.execute(
            "INSERT INTO tiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            duplicate,
        )
    (second / SHARD_MANIFEST_NAME).unlink()
    (second / SUCCESS_MARKER_NAME).unlink()
    _finalize_shard(second, campaign, 2001)

    with pytest.raises(ShardValidationError, match="conflicting metadata or content"):
        consolidate(tmp_path)
