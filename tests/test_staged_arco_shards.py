import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import budget, config, grid
from src.specs import DataSourceConfig, DomainRequest, SurfaceBehaviour
from src_arco import cache, selection
from src_arco.campaign import Campaign, CampaignConfigError, initialize_campaign
from src_arco.consolidation import consolidate, finalize_year_shard, year_shard_root
from src_arco.legacy_migration import migrate_legacy_year
from src_arco.shard_artifacts import (
    SUCCESS_MARKER_NAME,
    ShardValidationError,
)


def _tile_id_from_store(store: str | Path) -> str:
    name = Path(store).name
    if name.startswith(".") and name.endswith(".tmp.zarr"):
        return name.split(".", 2)[1]
    return name.removesuffix(".zarr")


@pytest.fixture(autouse=True)
def _in_memory_zarr(monkeypatch):
    registry: dict[str, xr.Dataset] = {}

    def fake_to_zarr(dataset, store, *args, **kwargs):
        path = Path(store)
        path.mkdir(parents=True, exist_ok=True)
        (path / "zarr.json").write_text("{}\n", encoding="utf-8")
        (path / "synthetic-data.bin").write_bytes(b"synthetic staged data")
        registry[_tile_id_from_store(path)] = dataset.copy(deep=True)

    def fake_open_zarr(store, *args, **kwargs):
        tile_id = _tile_id_from_store(store)
        if tile_id not in registry:
            raise FileNotFoundError(f"Unknown synthetic Zarr store: {store}")
        return registry[tile_id].copy(deep=True)

    monkeypatch.setattr(xr.Dataset, "to_zarr", fake_to_zarr)
    monkeypatch.setattr(xr, "open_zarr", fake_open_zarr)
    monkeypatch.setattr(
        cache,
        "_normalize_zarr_chunks",
        lambda dataset, source_cfg: dataset,
    )
    return registry


def _campaign(*, start_year: int = 1940, end_year: int = 1940) -> Campaign:
    return Campaign(
        campaign_id="test-campaign",
        start_year=start_year,
        end_year=end_year,
        start_month_day="06-01",
        end_month_day="06-01",
        region=None,
        bbox=(1.0, 5.0, 11.0, 15.0),
        margin_n=1,
        zg_top_pa=80000.0,
        zg_bottom="pressure_level",
        zg_bottom_pa=100000.0,
        allow_bottom_overflow=False,
        time_chunk="day",
        attempt_timeout_seconds=300.0,
        include_benchmark_variables=False,
    )


def _request() -> DomainRequest:
    return DomainRequest(
        bbox=(1.0, 5.0, 11.0, 15.0),
        margin_n=1,
        zg_top_pressure=80000.0,
        zg_bottom="pressure_level",
        zg_bottom_pressure=100000.0,
    )


def _source_cfg(year: int = 1940) -> DataSourceConfig:
    return DataSourceConfig(
        kind="arco_era5",
        arco_path=config.DEFAULT_ARCO_PATH,
        time_start=f"{year:04d}-06-01T00:00:00",
        time_end=f"{year:04d}-06-01T23:00:00",
    )


def _staged_cfg(root: Path, year: int = 1940) -> DataSourceConfig:
    return DataSourceConfig(
        kind="staged_arco_cache",
        staged_cache_root=str(root),
        arco_path=config.DEFAULT_ARCO_PATH,
        time_start=f"{year:04d}-06-01T00:00:00",
        time_end=f"{year:04d}-06-01T23:00:00",
    )


def _canonical_dataset(year: int = 1940) -> xr.Dataset:
    time = np.arange(
        np.datetime64(f"{year:04d}-06-01T00:00:00"),
        np.datetime64(f"{year:04d}-06-02T00:00:00"),
        np.timedelta64(1, "h"),
    )
    level = np.array([100000.0, 90000.0, 80000.0, 70000.0])
    lat = np.arange(0.0, 7.0)
    lon = np.arange(10.0, 17.0)
    shape_4d = (time.size, level.size, lat.size, lon.size)
    shape_3d = (time.size, lat.size, lon.size)
    index = np.arange(np.prod(shape_4d), dtype=float).reshape(shape_4d)
    hour_signal = np.sin(np.arange(time.size) * 2.0 * np.pi / 24.0)
    values = index * 1.0e-5 + hour_signal[:, None, None, None]

    dataset = xr.Dataset(
        {
            "T": xr.DataArray(
                280.0 + values,
                dims=("time", "level", "lat", "lon"),
                attrs={"units": "K"},
            ),
            "u": xr.DataArray(
                1.0 + values,
                dims=("time", "level", "lat", "lon"),
            ),
            "v": xr.DataArray(
                2.0 + values,
                dims=("time", "level", "lat", "lon"),
            ),
            "w": xr.DataArray(
                0.01 + values * 1.0e-3,
                dims=("time", "level", "lat", "lon"),
            ),
            "sp": xr.DataArray(
                np.full(shape_3d, 101000.0),
                dims=("time", "lat", "lon"),
            ),
        },
        coords={"time": time, "level": level, "lat": lat, "lon": lon},
    )
    dataset["level"].attrs["units"] = "Pa"
    return dataset


def _write_year(
    root: Path,
    campaign: Campaign,
    year: int,
    *,
    git_commit: str = "a" * 40,
) -> Path:
    request = _request()
    source_cfg = _source_cfg(year)
    shard_root = year_shard_root(root, year)
    tile = cache.build_arco_cache_tile(
        _canonical_dataset(year),
        request,
        include_benchmark_variables=False,
    )
    cache.write_tile(
        shard_root,
        tile,
        source_cfg,
        request,
        include_benchmark_variables=False,
    )
    finalize_year_shard(
        root,
        year,
        pbs_job_id=f"{year}.venus",
        pbs_array_index=str(year - campaign.start_year),
        git_commit=git_commit,
    )
    return shard_root


def _budget_result(dataset: xr.Dataset, tmp_path: Path) -> xr.Dataset:
    request = _request()
    surface = SurfaceBehaviour(
        allow_bottom_overflow=False,
        use_surface_variables=False,
        surface_variable_mode="none",
    )
    domain, halo, spec = grid.determine_domain(dataset, request)
    return budget.calculate_budget(
        domain,
        halo,
        spec,
        surface,
        integral_diagnostics_flag=True,
        plot_dir=str(tmp_path),
        plot_flag=False,
    )


def test_campaign_initialization_is_idempotent_and_immutable(tmp_path):
    campaign = _campaign()
    root = tmp_path / campaign.campaign_id

    first = initialize_campaign(root, campaign)
    second = initialize_campaign(root, campaign)

    assert first == second == root / "campaign.json"
    assert Campaign.from_file(first) == campaign

    changed = Campaign.from_mapping(
        {
            **campaign.to_mapping(),
            "staging": {
                **campaign.to_mapping()["staging"],
                "time_chunk": "month",
            },
        }
    )
    with pytest.raises(CampaignConfigError, match="different configuration"):
        initialize_campaign(root, changed)


def test_campaign_initialization_rejects_legacy_cache_root(tmp_path):
    campaign = _campaign()
    root = tmp_path / campaign.campaign_id
    cache.init_cache_root(root)

    with pytest.raises(CampaignConfigError, match="legacy cache root"):
        initialize_campaign(root, campaign)


def test_legacy_migration_preserves_dataset_through_consolidation(tmp_path):
    campaign = _campaign()
    legacy_root = tmp_path / "legacy"
    request = _request()
    source_cfg = _source_cfg()
    tile = cache.build_arco_cache_tile(
        _canonical_dataset(),
        request,
        include_benchmark_variables=False,
    )
    cache.write_tile(
        legacy_root,
        tile,
        source_cfg,
        request,
        include_benchmark_variables=False,
    )

    campaign_root = tmp_path / campaign.campaign_id
    initialize_campaign(campaign_root, campaign)
    migrate_legacy_year(
        legacy_root,
        campaign_root,
        1940,
        git_commit="a" * 40,
    )
    summary = consolidate(campaign_root)

    assert summary["year_count"] == 1
    assert summary["tile_count"] == 1
    migrated = cache.load_cache_dataset(
        campaign_root,
        _staged_cfg(campaign_root),
        request,
    )
    legacy = cache.load_cache_dataset(
        legacy_root,
        _staged_cfg(legacy_root),
        request,
    )
    xrt.assert_identical(migrated, legacy)


def test_consolidated_shard_preserves_dataset_and_budget_physics(tmp_path):
    campaign = _campaign()
    campaign_root = tmp_path / campaign.campaign_id
    initialize_campaign(campaign_root, campaign)
    shard_root = _write_year(campaign_root, campaign, 1940)

    summary = consolidate(campaign_root)

    assert summary["year_count"] == 1
    assert summary["tile_count"] == 1
    assert (campaign_root / "cache.sqlite").is_file()
    assert (campaign_root / "consolidation.json").is_file()
    assert (shard_root / SUCCESS_MARKER_NAME).is_file()

    with sqlite3.connect(campaign_root / "cache.sqlite") as connection:
        path = connection.execute("SELECT path FROM tiles").fetchone()[0]
    assert path.startswith("shards/year=1940/tiles/")

    consolidated = cache.load_cache_dataset(
        campaign_root,
        _staged_cfg(campaign_root),
        _request(),
    )

    legacy_root = tmp_path / "legacy"
    tile = cache.build_arco_cache_tile(
        _canonical_dataset(),
        _request(),
        include_benchmark_variables=False,
    )
    cache.write_tile(
        legacy_root,
        tile,
        _source_cfg(),
        _request(),
        include_benchmark_variables=False,
    )
    legacy = cache.load_cache_dataset(
        legacy_root,
        _staged_cfg(legacy_root),
        _request(),
    )

    xrt.assert_identical(consolidated, legacy)
    consolidated_budget_input = consolidated.isel(time=slice(0, 5)).load()
    legacy_budget_input = legacy.isel(time=slice(0, 5)).load()
    xrt.assert_allclose(
        _budget_result(
            consolidated_budget_input,
            tmp_path / "consolidated-budget",
        ),
        _budget_result(legacy_budget_input, tmp_path / "legacy-budget"),
    )


def test_finalize_rejects_incomplete_hourly_coverage(tmp_path):
    campaign = _campaign()
    root = tmp_path / campaign.campaign_id
    initialize_campaign(root, campaign)
    shard_root = year_shard_root(root, 1940)
    truncated = _canonical_dataset().isel(time=slice(0, -1))
    source_cfg = DataSourceConfig(
        kind="arco_era5",
        arco_path=config.DEFAULT_ARCO_PATH,
        time_start="1940-06-01T00:00:00",
        time_end="1940-06-01T22:00:00",
    )
    tile = cache.build_arco_cache_tile(
        truncated,
        _request(),
        include_benchmark_variables=False,
    )
    cache.write_tile(
        shard_root,
        tile,
        source_cfg,
        _request(),
        include_benchmark_variables=False,
    )

    with pytest.raises(ShardValidationError, match="missing 1 hour"):
        finalize_year_shard(root, 1940)
    assert not (shard_root / SUCCESS_MARKER_NAME).exists()


def test_consolidation_requires_every_campaign_year(tmp_path):
    campaign = _campaign(start_year=1940, end_year=1941)
    root = tmp_path / campaign.campaign_id
    initialize_campaign(root, campaign)
    _write_year(root, campaign, 1940)

    with pytest.raises(ShardValidationError, match="Missing shard directory for year=1941"):
        consolidate(root)
    assert not (root / "cache.sqlite").exists()


def test_consolidation_combines_isolated_year_folders(tmp_path):
    campaign = _campaign(start_year=1940, end_year=1941)
    root = tmp_path / campaign.campaign_id
    initialize_campaign(root, campaign)
    _write_year(root, campaign, 1940)
    _write_year(root, campaign, 1941)

    summary = consolidate(root)

    assert summary["year_count"] == 2
    assert summary["tile_count"] == 2
    assert summary["git_commit"] == "a" * 40
    assert [shard["year"] for shard in summary["shards"]] == [1940, 1941]
    with sqlite3.connect(root / "cache.sqlite") as connection:
        paths = [
            row[0]
            for row in connection.execute("SELECT path FROM tiles ORDER BY path")
        ]
    assert any(path.startswith("shards/year=1940/") for path in paths)
    assert any(path.startswith("shards/year=1941/") for path in paths)


def test_consolidation_rejects_mixed_producer_commits(tmp_path):
    campaign = _campaign(start_year=1940, end_year=1941)
    root = tmp_path / campaign.campaign_id
    initialize_campaign(root, campaign)
    _write_year(root, campaign, 1940, git_commit="a" * 40)
    _write_year(root, campaign, 1941, git_commit="b" * 40)

    with pytest.raises(ShardValidationError, match="not campaign commit"):
        consolidate(root)
    assert not (root / "cache.sqlite").exists()


def test_failed_reconsolidation_preserves_published_catalog(tmp_path):
    campaign = _campaign()
    root = tmp_path / campaign.campaign_id
    initialize_campaign(root, campaign)
    shard_root = _write_year(root, campaign, 1940)
    consolidate(root)
    database_before = (root / "cache.sqlite").read_bytes()

    success = json.loads((shard_root / SUCCESS_MARKER_NAME).read_text())
    success["manifest_sha256"] = "0" * 64
    (shard_root / SUCCESS_MARKER_NAME).write_text(
        json.dumps(success),
        encoding="utf-8",
    )

    with pytest.raises(ShardValidationError, match="manifest checksum"):
        consolidate(root)
    assert (root / "cache.sqlite").read_bytes() == database_before


def test_consolidation_rejects_tampered_shard_content(tmp_path):
    campaign = _campaign()
    root = tmp_path / campaign.campaign_id
    initialize_campaign(root, campaign)
    shard_root = _write_year(root, campaign, 1940)
    data_file = next(shard_root.rglob("synthetic-data.bin"))
    data_file.write_bytes(b"tampered! staged data")

    with pytest.raises(ShardValidationError, match="checksum mismatch"):
        consolidate(root)
    assert not (root / "cache.sqlite").exists()
