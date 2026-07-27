# Venus Production Staged ARCO Workflow

Production ARCO staging uses one immutable campaign directory and one isolated
cache shard per year. The default layout is:

```text
results/staged_arco_cache/<campaign-id>/
├── campaign.json
├── cache.sqlite
├── consolidation.json
└── shards/
    └── year=YYYY/
        ├── cache.sqlite
        ├── shard-manifest.json
        ├── _SUCCESS.json
        └── tiles/
```

Each array task owns one `year=YYYY` directory. No retrieval task writes the
campaign-level SQLite database. A dependent consolidation job validates every
year and atomically publishes that database after all retrieval tasks succeed.

## Configure a campaign

Edit or override the values in `production_run_cli_settings.sh`:

- `CAMPAIGN_ID`
- `START_YEAR` and `END_YEAR`
- `RUN_START_MONTH_DAY` and `RUN_END_MONTH_DAY`
- `REGION` and `MARGIN_N`
- `ZG_TOP_PA`, `ZG_BOTTOM`, and optional `ZG_BOTTOM_PA`
- `ALLOW_BOTTOM_OVERFLOW`
- `ENABLE_BENCHMARK_VARIABLES`
- `STAGED_CACHE_BASE_ROOT`

The default campaign cache root is
`${STAGED_CACHE_BASE_ROOT}/${CAMPAIGN_ID}`. A campaign ID is permanently bound
to its normalized `campaign.json`. Use a new campaign ID when any domain,
season, year, vertical-boundary, benchmark, chunking, or timeout setting
changes.

## Submit on Venus

Commit and push the exact production branch, then fast-forward the clean Venus
checkout. From that checkout:

```bash
PROJECT_ROOT=/absolute/path/to/eulerian_heat_budget \
  schedulers/submit_staged_arco_production.sh
```

The submission wrapper refuses a dirty or unpushed checkout. It submits:

1. the yearly retrieval array; and
2. `schedule_consolidate_staged_arco_cache.sh` with an `afterokarray`
   dependency on that array.

It prints both PBS job IDs. Record them with the campaign ID, Git commit, cache
root, and production configuration.

The retrieval jobs and consolidation job activate
`${VENUS_MAMBA_ENV:-dev_env}`. Retrieval requests 8 GB per yearly task.
Consolidation requests 4 GB and reads manifests and Zarr metadata
sequentially.

## Resume and consume

Resubmitting an unchanged campaign is safe. Existing exact tiles are skipped,
and completed shards must still pass checksum, coverage, campaign, and Git
validation. Incomplete shards remain isolated from the campaign catalog.

Do not start production heat-budget jobs until the consolidation job succeeds.
The calculation uses the campaign root:

```text
--data-source staged_arco_cache \
--staged-cache-root /path/to/<campaign-id>
```

The consolidated catalog points into the yearly shard directories. The
canonical staged dataset and heat-budget calculation are unchanged by the
storage layout.
