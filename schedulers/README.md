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
point into a Git checkout. Every Venus production checkout must be
commit-specific, live below `~/eulerian-heat-budget/production/`, use the named
`production_development_staged` branch, and track
`origin/production_development_staged`. It must not change branch or commit
while any associated PBS job is queued or running. Development checkouts remain
independent and may be changed without affecting production.

Production ARCO staging uses one immutable campaign directory and one isolated
cache shard per year:

```text
campaign-data/staged-zarr/<region>/<campaign-id>/
├── campaign.json
├── production_run.json
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

Before `qsub`, the submission wrapper atomically creates or validates
`production_run.json`. It records the normalized campaign settings and hash,
ARCO source, authoritative Git branch and commit, checkout and output paths,
Mamba environment, array shape, requested resources, and an appendable history
of retrieval and consolidation job IDs. Resubmission is accepted only when the
stored production identity is unchanged.

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

Integrate and push every intended production change onto
`production_development_staged`, then create or fast-forward a clean,
commit-specific Venus checkout below `~/eulerian-heat-budget/production/`.
From that checkout:

```bash
PROJECT_ROOT=/home/USER/eulerian-heat-budget/production/eulerian_heat_budget-COMMIT \
  schedulers/submit_staged_arco_production.sh
```

The submission wrapper queries the live remote and requires all of these
conditions without an override:

- the checkout is below `${EHB_WORKSPACE_ROOT}/production/`;
- the named branch is exactly `production_development_staged`;
- its upstream is exactly `origin/production_development_staged`;
- the checkout is clean; and
- `HEAD` equals the live remote branch tip.

An otherwise valid feature commit ahead of `production_development_staged` is
intentionally rejected. Merge, validate, and push it to the authoritative
branch first. Google Batch and Alliance Slurm use their separate authoritative
branches and submission workflows; they do not bypass this Venus preflight.

After preflight, the wrapper submits:

1. the yearly retrieval array; and
2. `schedule_consolidate_staged_arco_cache.sh` with an `afterok`
   dependency on that array.

It prints both PBS job IDs. Record them with the campaign ID, Git commit, cache
root, log directory, and production configuration. The wrapper also records
both IDs in campaign-level `production_run.json`. It creates
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
PROJECT_ROOT=/home/USER/eulerian-heat-budget/production/eulerian_heat_budget-COMMIT \
  schedulers/submit_run_budget_production.sh
```

The default result directory is
`campaign-data/run-budget/<region>/<run-id>/`.

## Migrate a legacy indexed cache

Use `submit_legacy_staged_arco_migration.sh` when an existing legacy
`cache.sqlite` and `tiles/` collection must be reorganized into yearly shards
without downloading the ARCO data again. The migration:

- treats the legacy cache as read-only;
- creates hard links for tile files, so source and destination must be on the
  same filesystem;
- creates one private SQLite catalog per year;
- runs the ordinary coverage, campaign, checksum, and Zarr validation;
- writes `shard-manifest.json` and `_SUCCESS.json` last; and
- optionally runs the ordinary campaign consolidation as a dependent job.

The migrated campaign intentionally has no `production_run.json`. Its
provenance is recorded in `campaign.json`, yearly success markers, shard
manifests, PBS logs, and `consolidation.json`.

Set the ordinary campaign variables plus `LEGACY_CACHE_ROOT`. For example, a
single-year smoke migration can use `TASK_RANGE=0-0` and
`SUBMIT_CONSOLIDATION=0`. After validation, submit the remaining task range
with consolidation enabled. Never point the destination at the legacy source.
