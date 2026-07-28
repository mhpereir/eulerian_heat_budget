# Venus direct production workflow

Direct production runs read the public ARCO ERA5 Zarr store and compute the
Eulerian heat budget without writing an intermediate staged cache.

On Venus, keep immutable code checkouts and campaign data separate:

```text
~/eulerian-heat-budget/
├── production/
│   └── eulerian_heat_budget-<commit>/
└── campaign-data/
    ├── run-budget/<region>/<run-id>/
    └── logs/<region>/<run-id>/
```

The submission wrapper requires a clean `production_development` checkout
below `production/`, verifies that its commit is the live remote branch tip,
and exports that commit to every PBS task. Runtime tasks refuse a different
commit or a dirty checkout.

Review `production_run_cli_settings.sh`, then submit from Venus:

```bash
PROJECT_ROOT="$HOME/eulerian-heat-budget/production/eulerian_heat_budget-<commit>" \
  "$PROJECT_ROOT/schedulers/submit_run_budget_production.sh"
```

The defaults describe the `pnw_hotz` May 1 to October 31, 1940-2025 rerun,
with a surface-pressure lower boundary, a 700 hPa top, and at most five
concurrent yearly tasks. The wrapper rejects any data source other than
`arco_era5`.
