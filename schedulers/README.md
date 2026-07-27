# Venus Production Staged ARCO Workflow

Production code and production data have separate lifecycles on Venus. The
default workspace is:

```text
~/eulerian-heat-budget/
├── development/
│   └── <editable-git-checkout>/
├── production/
│   └── <commit-specific-git-checkout>/
└── campaign-data/
    ├── staged-zarr/
    │   └── <region>/<campaign-id>/
    ├── run-budget/
    │   └── <region>/<run-id>/
    └── logs/
        └── <region>/<run-id>/
```

`campaign-data` may be a symlink to backed-up project storage. It must never
point into a Git checkout. Production checkouts are commit-specific and must
not change branch or commit while any associated PBS job is queued or running.
Development checkouts remain independent and may be changed without affecting
production.

Production ARCO staging uses one immutable campaign directory and one isolated
cache shard per year:

```text
campaign-data/staged-zarr/<region>/<campaign-id>/
├── campaign.json
├── cache.sqlite
├── consolidation.json
└── shards/year=YYYY/
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
- `EHB_WORKSPACE_ROOT` and `EHB_CAMPAIGN_DATA_ROOT`
- `EHB_STAGED_ZARR_ROOT`, `EHB_RUN_BUDGET_ROOT`, and `EHB_LOG_ROOT`
- `STAGED_CACHE_BASE_ROOT`
- `RUN_ID` and `PRODUCTION_OUTPUT_DIR`

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
2. `schedule_consolidate_staged_arco_cache.sh` with an `afterok`
   dependency on that array.

It prints both PBS job IDs. Record them with the campaign ID, Git commit, cache
root, log directory, and production configuration. The wrapper creates
`${LOG_DIR}` under `campaign-data/logs/<region>/<run-id>/` and passes it to
`qsub -o`, so PBS spool output and the scripts' detailed `tee` logs remain
outside the Git checkout and the submission shell's working directory.

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

Submit the heat-budget array through its wrapper so the commit, external data
paths, and PBS log destination are verified:

```bash
PROJECT_ROOT=/absolute/path/to/commit-specific-checkout \
  schedulers/submit_run_budget_production.sh
```

The default result directory is
`campaign-data/run-budget/<region>/<run-id>/`.
