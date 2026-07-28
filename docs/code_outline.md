# Eulerian Heat Budget - Shared Code Outline

July 27, 2026

## 1. Purpose And Scope

This document is the code-faithful architectural reference for the Eulerian
heat-budget workflow in pressure coordinates. It describes the shared
scientific calculation, input adapters, runtime modes, data contracts, output
schema, and test coverage represented across these active branch tips:

- `production_development`
- `production_development_staged`
- `drac_development_2_staged`
- `google_development_staged`

The branch tips have different scheduler, data-retrieval, deployment, region,
and machine-path adapters. Those differences must remain outside the shared
heat-budget calculation. Once an input adapter has returned the canonical
dataset, `scripts/run_budget.py`, `src/budget.py`, and the scientific modules
use the same calculation and output contracts.

The implementation computes volume-integrated temperature storage,
temperature-anomaly advection, adiabatic heating, and a residual diabatic term
over a regional control volume. It also preserves mass-closure, geometric,
per-face, and benchmark diagnostics needed by downstream analysis.

`docs/Eulerian Heat Budget - Reformulation.md` is the source of truth for the
physics. This document records how those definitions are represented in code.
`docs/full-column-benchmarking-strategy.md` defines the ERA5 comparison
hierarchy, hard-data fixture contract, and scientific tolerance policy.

## 2. Physical And Sign Contracts

### 2.1 Heat Advection

Heat advection is positive when heat is carried into the control volume.
Equivalently, if `n` is the outward unit normal,

```text
A = - integral_A T' (U dot n) dA
```

where

```text
T' = T - T_domain_avg
```

The signed `advection_term` and every `flux_contribution_<face>` use this
positive-into-domain convention:

- inward transport is positive on every face
- outward transport is negative on every face
- for positive `u`, `v`, and `w`, the face multipliers are positive on west,
  south, and top and negative on east, north, and bottom
- reversing a face-normal velocity reverses its signed contribution

The fixed-pressure bottom face is present only when
`DomainSpec.zg_bottom == "pressure_level"`. A surface-pressure bottom is a
moving material boundary and does not produce a separate bottom advective
face.

### 2.2 Diabatic Residual

The code defines

```text
S = A + C + D
```

with:

- `S = dT_dt`, the domain-volume-scaled tendency of `T_domain_avg`
- `A = advection_term`, positive into the domain
- `C = adiabatic_term`
- `D = diabatic_term`

The diagnosed diabatic residual is therefore:

```text
D = S - A - C
```

`src/terms.py::compute_diabatic_term()` implements this expression, and
`src/budget.py::calculate_budget()` passes `dT_dt`, `advection_term`, and
`adiabatic_term` in that order.

### 2.3 Mass Closure And Heat-Error Diagnostics

`net_mass_advection` is positive into the domain. The implemented mass-closure
residual is:

```text
delta_M = net_mass_advection - dV_dt
```

Positive `delta_M` means that the calculated net inflow exceeds the calculated
rate of control-volume growth.

The code retains two heat-scaled forms for downstream analysis:

```text
residual_heat  = delta_M * T_domain_avg
advection_error = delta_M * T_scale
```

`T_scale` is the root-mean-square temperature-anomaly scale computed from
`T - T_domain_avg`. `residual_heat` is a diagnostic output and is not added to
the diabatic residual. `advection_error` is used as the current
advection-related uncertainty scale in result plots.

### 2.4 Benchmark Sign Normalization

ARCO benchmark fluxes are converted to the output-file convention before they
are merged into the budget result. All `benchmark_mass_flux_*` and
`benchmark_heat_flux_*` output fields are positive into the domain, including
their per-face values.

## 3. ERA5 Data-Access Modes

The shared calculation supports three data-access patterns. Backend launchers
may arrange files or queue retrieval differently, but they must select one of
these adapters and produce the canonical dataset described in Section 7.

### 3.1 Pre-downloaded ERA5

Data-source kind:

```text
local_era5
```

This path reads component NetCDF files from a configured local directory:

- `T.nc`
- `ux.nc`
- `uy.nc`
- `uz.nc`
- `sfp.nc`

When surface variables are enabled, it also reads:

- `surface_temperature.nc`
- `surface_ux.nc`
- `surface_uy.nc`

On the portable branch, the directory comes from `--local-data-path` or
`EHB_DATA_ROOT`. Backend tips may inject the same path through their local
configuration or scheduler wrapper. Machine-specific paths are not part of the
scientific contract.

### 3.2 Streamed ARCO ERA5

Data-source kind:

```text
arco_era5
```

This path opens the public ARCO ERA5 Zarr store and reads the requested
variables, time range, and spatial domain through xarray and GCS-compatible
storage. It supports retry with exponential backoff for recognized transient
open failures.

Core ARCO fields are renamed to the internal schema:

```text
temperature                 -> T
u_component_of_wind         -> u
v_component_of_wind         -> v
vertical_velocity           -> w
surface_pressure            -> sp
```

Optional surface variables can also be streamed when requested. The benchmark
add-on streams vertically integrated mass and heat fluxes plus `vithe`,
`viec`, and `vithed`. The three heating diagnostics are valid only for a
1 hPa-to-surface-pressure calculation.

### 3.3 Staged ARCO Zarr Cache

Data-source kind on the staged branch tips:

```text
staged_arco_cache
```

This is a two-phase workflow:

1. `scripts/staged_arco_retrieval.py` streams selected ARCO data and writes
   compact local Zarr tiles.
2. `scripts/run_budget.py` reads those tiles without requiring ARCO or other
   network access.

The cache is indexed by `cache.sqlite` and stores tiles under `tiles/`. Cache
selection validates source identity, requested time coverage, horizontal and
vertical coverage, benchmark availability, and the wall stencils required by
the budget calculation. Complementary time tiles can be combined when their
coverage is complete and consistent.

Production staging uses an isolated campaign and shard layout:

```text
<staged-cache-base>/<campaign-id>/
├── campaign.json
├── cache.sqlite                 # written atomically by consolidation
├── consolidation.json
└── shards/
    └── year=YYYY/
        ├── cache.sqlite         # private to one retrieval task
        ├── shard-manifest.json
        ├── _SUCCESS.json
        └── tiles/
            └── <tile-id>.zarr/
```

Every yearly scheduler task writes only its own shard database and Zarr
directory. It validates complete hourly coverage, records checksums and Git
provenance, and writes `_SUCCESS.json` last. A dependent consolidation job
requires every campaign year, verifies file hashes, cache schemas, campaign
parameters, source metadata, time coordinates, and a common producer commit,
then atomically replaces the campaign-level `cache.sqlite`. Its catalog paths
refer into `shards/year=YYYY/`, so the ordinary staged-cache loader requires no
scientific or data-schema special case.

The current staged cache schema stores:

- full selected `T`, `w`, and `sp`
- compact `u_wall` and `v_wall` arrays containing the required lateral
  velocity stencils
- optional compact benchmark wall fluxes
- optional full-horizontal `vithe`, `viec`, and `vithed` benchmark fields
- pressure bounds and cache metadata

`src_arco.selection` reconstructs canonical `u` and `v` arrays before the
dataset enters the shared validation and budget pipeline. Interior velocity
locations that are not required by the wall-flux calculation can remain
missing.

Current staged-cache constraints:

- `--staged-cache-root` is required
- staged surface variables `T2m`, `u10`, and `v10` are not implemented
- benchmark variables must have been included during staging before
  `--include-benchmark-variables` can be used offline

Staging and consolidation can be launched by PBS, Slurm, or Google adapters.
Those queueing and transfer mechanisms do not change the cache or heat-budget
contracts.

The staged retrieval path opens the global ARCO store with `chunks=None` so
xarray does not construct a store-wide Dask graph. It applies the requested
time, horizontal, vertical, and wall-only selections first, materializes the
selected variables one at a time, and applies the configured Dask/Zarr chunks
only to the compact tile being written.

### 3.4 Data-Source Selection

`--data-source` is explicit in the active `run_budget` implementations. The
legacy `DEFAULT_DATA_SOURCE` constant exists on the branch tips but is not used
to replace an omitted CLI value. An unsupported or omitted source therefore
fails during `DataSourceConfig` construction rather than silently selecting a
different access method.

## 4. Runtime Pipelines

### 4.1 Shared Heat-Budget Calculation

After CLI and output-mode resolution, the calculation flow is:

1. `src/cli.py` parses domain, surface, source, time, diagnostic, and output
   options.
2. `scripts/run_budget.py` constructs:
   - `specs.DomainRequest`
   - `specs.SurfaceBehaviour`
   - `specs.DataSourceConfig`
3. `src/io.py` dispatches to the selected ERA5 adapter and standardizes its
   result.
4. `src/validate.py` enforces the canonical loaded-dataset schema.
5. Optional full-column benchmark inputs are loaded from streamed ARCO or the
   staged cache.
6. `src/grid.py::determine_domain()` returns:
   - `ds_domain`
   - `ds_halo`
   - `DomainSpec`
7. `src/grid.py` constructs horizontal areas, vertical wall areas, line
   elements, and pressure-coordinate cell volumes.
8. `src/weights.py` constructs horizontal-face, vertical-wall, and volume
   occupancy weights.
9. `src/terms.py` computes storage, volume, temperature-average, advection,
   adiabatic, mass-closure, and benchmark terms.
10. `src/budget.py::calculate_budget()` assembles and computes the output
    dataset.
11. Diagnostic and summary plots are written only when diagnostic plotting is
    enabled.
12. A second constant-temperature calculation is run only when the
    constant-temperature test is enabled.

The scientific path from canonical dataset through `calculate_budget()` is
independent of the scheduler and data-retrieval backend.

### 4.2 Ad Hoc Runs

For an ad hoc run:

- `src/run_outputs.py::prepare_run_paths()` creates a unique run directory
- `run_info.json` records request, source, resolved domain, surface behavior,
  CLI arguments, scheduler context, and git provenance
- plots are optional
- NetCDF output is optional through `--write-netcdf`
- the optional constant-temperature NetCDF result is stored separately
- `--overwrite-output` controls explicit replacement of existing NetCDF output

The default portable output root comes from `--output-root`,
`EHB_OUTPUT_ROOT`, or the repository-local `results/` fallback.

### 4.3 Production Manifest Initialization

`--init-production-manifest` with `--production-output-dir`,
`--production-start-year`, and `--production-end-year` writes
`production_run.json` and exits before loading ERA5 data.

The manifest records campaign-wide request, source, surface behavior, output
layout, scheduler context, CLI arguments, and git provenance.

### 4.4 Production Year Runs

A production calculation:

- requires an existing `production_run.json`
- requires `time_start` and `time_end` within one calendar year
- writes `annual/heat_budget_<year>.nc`
- uses `plots/<year>/` when plots are enabled
- refuses to replace existing output unless overwrite was explicitly allowed
- does not write an ad hoc `run_info.json` for each year

Queue-array construction and campaign submission remain backend-specific.

### 4.5 Dask Client

The executable entrypoint parses arguments before or as part of starting a
local Dask distributed client, depending on the branch-tip wrapper. The shared
worker settings are currently:

- `n_workers=4`
- `threads_per_worker=1`
- `processes=True`
- `memory_limit="8GB"`

The portable entrypoint closes the client through a context manager. Backend
tips may rely on process termination after `main()` returns.

## 5. Repository Structure

The following tree is the union of the shared implementation and the active
staged/backend additions. Items marked as branch-specific are not expected on
every tip.

```text
eulerian_heat_budget/
├── .dockerignore                       # Google staged tip
├── AGENTS.md
├── Dockerfile                          # Google staged tip
├── README.md
├── environment.yml
├── pytest.ini
├── requirements-arco.in                # Google staged tip
├── requirements-arco.txt               # Google staged tip
├── archive/
│   └── scripts/
│       └── update_run_catalog.py
├── bugs/
│   └── fixed/
│       └── staged_arco_cache_vertical_tile_bug_report.md
├── cache_viz/
│   └── visualize_staged_arco_cache.py
├── deployment/                         # backend-specific
│   └── gcp/                            # Google staged tip
│       ├── README.md
│       ├── build_arco_image.sh
│       ├── campaign.example.json
│       ├── campaign.py
│       ├── render_batch_job.py
│       ├── run_yearly_retrieval.py
│       ├── setup_batch_resources.sh
│       ├── shard_artifacts.py
│       └── submit_arco_batch.sh
├── docs/
│   ├── Eulerian Heat Budget - Reformulation.md
│   ├── README.md                        # Google staged tip
│   ├── code_outline.md
│   └── google-cloud-batch-deployment.md # Google staged tip
├── notebooks/
│   └── check_pbl_colab.ipynb
├── schedulers/
│   ├── production_run_cli_settings.sh   # staged tips
│   ├── schedule_run_budget.sh
│   ├── schedule_run_budget_production.sh
│   ├── schedule_run_budget*_slurm.sh    # DRAC staged tip
│   ├── schedule_consolidate_staged_arco_cache.sh
│   ├── schedule_staged_arco_retrieval*.sh
│   ├── schedule_staged_arco_retrieval*_slurm.sh
│   ├── submit_staged_arco_production.sh
│   └── single_run_cli_settings          # staged tips
├── scripts/
│   ├── closure_testing.py               # empty placeholder on some tips
│   ├── consolidate_staged_arco_cache.py # sharded staged tips
│   ├── run_budget.py
│   ├── staged_arco_campaign.py
│   └── staged_arco_retrieval.py         # staged tips
├── src/
│   ├── __init__.py
│   ├── budget.py
│   ├── cli.py
│   ├── config.py
│   ├── grid.py
│   ├── integrals.py
│   ├── io.py
│   ├── plot_diagnostics.py
│   ├── plot_results.py
│   ├── run_outputs.py
│   ├── specs.py
│   ├── terms.py
│   ├── validate.py
│   └── weights.py
├── src_arco/                           # staged tips
│   ├── __init__.py
│   ├── cache.py
│   ├── campaign.py
│   ├── consolidation.py
│   ├── selection.py
│   ├── shard_artifacts.py
│   └── variables.py
└── tests/
    ├── test_arco_cache.py              # staged tips
    ├── test_benchmark_diagnostics.py
    ├── test_budget_closure.py
    ├── test_gcp_batch_deployment.py    # Google staged tip
    ├── test_grid.py
    ├── test_integrals.py
    ├── test_io.py
    ├── test_run_budget.py
    ├── test_run_outputs.py
    ├── test_staged_arco_shards.py        # sharded staged tips
    ├── test_update_run_catalog.py
    └── test_weights.py
```

Generated data, caches, credentials, logs, and calculation results are not part
of the implementation contract and should remain outside version control.

## 6. Module Responsibilities

### `src/config.py`

Defines physical constants and branch-local runtime defaults:

- physical constants: `g`, `R_value`, `R_earth`, `cp`
- local and ARCO paths and tokens
- ARCO retry and staging timeout controls where supported
- time-window defaults
- named region bounding boxes
- margin and vertical-boundary defaults
- surface-behavior defaults
- output-root or plot-root defaults
- Dask chunk sizes
- default plotting and constant-temperature controls

Branch tips may carry different regions and filesystem defaults. These are
configuration differences, not changes to the scientific API.

### `src/specs.py`

Defines the immutable dataclass contracts:

- `DataSourceConfig`
  - `local_era5`
  - `arco_era5`
  - `staged_arco_cache` on staged tips
  - paths, token, chunks, staged cache root, and time window
- `DomainRequest`
  - requested bounding box
  - cell margin
  - top pressure
  - bottom-boundary mode and optional fixed pressure
- `SurfaceBehaviour`
  - bottom-overflow behavior
  - optional surface-variable behavior
- `DomainSpec`
  - resolved horizontal and vertical control-volume boundaries
  - validation for bottom-boundary consistency

### `src/cli.py`

Builds the shared heat-budget parser. Across the active tips it covers:

- named or explicit horizontal domains
- margin and vertical boundaries
- bottom-overflow and surface-variable behavior
- explicit data-source selection
- local input and staged-cache paths where supported
- time selection
- benchmark diagnostics
- diagnostic plots
- constant-temperature testing
- ad hoc NetCDF output
- production output and manifest initialization
- explicit output overwrite

Staging scripts extend this parser with staging-only controls such as campaign
season bounds, time-chunk size, and staging-attempt timeout.

### `src/io.py`

Owns the boundary between external data formats and the canonical dataset.
Responsibilities include:

- dispatching among pre-downloaded, streamed ARCO, and staged-cache adapters
- loading local ERA5 component files
- opening ARCO Zarr with transient-failure retry
- loading streamed full-column benchmark fields
- loading staged core and benchmark datasets where available
- renaming coordinates and variables
- dropping non-contract auxiliary coordinates
- converting pressure levels to Pa
- converting temperatures to Kelvin
- enforcing coordinate order and dimension order
- normalizing longitudes to `[-180, 180]`
- applying the requested time slice
- applying Dask chunks

The staged-tip signature also receives `DomainRequest`, because offline cache
coverage and wall reconstruction are request-dependent.

### `src_arco/variables.py`

Present on staged tips. Defines shared maps for ARCO core and benchmark
variables and enforces the current staged-cache restriction on surface
variables.

### `src_arco/selection.py`

Present on staged tips. Selects staging extents, derives pressure bounds,
builds compact wall-only tiles, computes required wall stencils, and
reconstructs canonical core and benchmark datasets.

### `src_arco/cache.py`

Present on staged tips. Owns the indexed Zarr cache:

- initializes the cache root and SQLite catalog
- identifies tiles from normalized source and request metadata
- writes tiles through temporary stores and a cache write lock
- registers tile coverage
- finds exact, covering, or complementary time tiles
- validates time, horizontal, vertical, wall, and benchmark coverage
- raises `OfflineCoverageError` when a request cannot be satisfied
- reconstructs the dataset without contacting ARCO

### `src_arco/campaign.py`

Present on sharded staged tips. Defines and validates the backend-neutral
campaign contract, canonical JSON and identity hash, named or explicit domain,
season, year range, staging settings, and immutable `campaign.json`
initialization.

### `src_arco/shard_artifacts.py`

Present on sharded staged tips. Validates shard SQLite schemas and safe relative
paths, builds per-file size and SHA-256 manifests, writes metadata atomically,
validates `_SUCCESS.json`, and computes content signatures for duplicate-tile
checks.

### `src_arco/consolidation.py`

Present on sharded staged tips. Validates yearly tile metadata and complete
hourly coverage, finalizes individual shards, requires all campaign years and a
single producer Git commit, rewrites tile paths into the campaign layout, and
atomically publishes the combined cache catalog and consolidation summary.

### `src/validate.py`

Strictly validates the canonical loaded dataset:

- required dimensions and variables
- required variable dimension order
- one-dimensional coordinates
- non-empty required dimensions
- strictly decreasing pressure levels
- strictly increasing latitude and longitude
- temperature in Kelvin
- regular time spacing

It raises errors instead of coercing data after the I/O boundary.

### `src/grid.py`

Resolves the domain and provides geometric operators:

- `determine_domain()`
  - treats input latitude and longitude as cell centers
  - selects centers inside the requested bounding box
  - applies `margin_n`
  - constructs `ds_domain` and a one-cell `ds_halo`
  - attaches horizontal and pressure bounds and cell IDs
  - returns the resolved `DomainSpec`
- `crop_to_target_grid()`
  - aligns benchmark data to a target grid
- `get_boundary_line_elements()`
  - returns lateral boundary line elements
- `get_horizontal_cell_areas()`
  - returns `A_horizontal(lat, lon)` in `m2`
- `get_vertical_cell_areas()`
  - returns `A_east`, `A_west`, `A_south`, and `A_north` in `m*Pa`
- `get_cell_volumes()`
  - returns `V_cell(level, lat, lon)` in `m2*Pa`

`margin_n` must be at least one because the halo is built using
`margin_n - 1`.

### `src/weights.py`

Builds occupancy weights:

- `area_weights_horizontal()`
  - always returns binary `W_top`
  - returns binary `W_bottom` for a fixed-pressure bottom
- `area_weights_vertical()`
  - returns fractional `W_east`, `W_west`, `W_south`, and `W_north`
- `volume_weights()`
  - returns fractional `W_volume`

Vertical and volume weights support both bottom-boundary modes. When
`allow_bottom_overflow=True` with a surface-pressure bottom, the lowest-layer
weight may exceed one to represent pressure depth below the lowest grid-cell
edge.

### `src/integrals.py`

Contains pure xarray integration operators:

- `area_integral()`
- `volume_integral_pcoords()`

They multiply field, geometry, and occupancy weights and sum all non-time
dimensions. Time differencing belongs to `src/terms.py`.

### `src/terms.py`

Computes physical and diagnostic terms:

- domain volume from weighted grid cells
- direct true domain volume from horizontal area and top/bottom pressures
- centered, forward, and backward time derivatives
- storage
- domain-average temperature
- face-centered advective inputs
- signed heat and mass advection
- adiabatic heating
- residual diabatic heating
- ARCO benchmark face fluxes
- benchmark aggregate and output diagnostics
- full-column `vithe`, `viec`, and `vithed` heating diagnostics

The active advection path subtracts `T_domain_avg` before constructing heat
fluxes. Optional surface fields are blended into the surface-adjacent layer
only when surface-variable use is enabled.

`compute_advective_term()` returns:

- `net_heat_advection`
- `flux_contribution_<face>`
- `net_mass_advection` when integral diagnostics are enabled
- `mass_flux_contribution_<face>` when integral diagnostics are enabled
- `abs_mass_advection_residual_fraction` when integral diagnostics are enabled

### `src/budget.py`

Orchestrates the calculation after data loading:

1. builds areas, volumes, and weights
2. computes storage and both domain-volume estimates
3. computes both volume tendencies
4. computes `T_domain_avg`, `dT_dt`, and `dT_dt_2`
5. subtracts `T_domain_avg` from the advection temperature inputs
6. prepares and integrates signed face fluxes
7. computes `T_scale`, `residual_heat`, and `advection_error`
8. computes adiabatic and residual diabatic terms
9. merges physical, closure, per-face, and optional benchmark outputs
10. computes the final xarray dataset
11. optionally writes diagnostic figures

The module does not choose a scheduler or retrieve staged data.

### `src/run_outputs.py`

Owns output paths, serialization, and provenance:

- scheduler-aware run IDs for PBS and Slurm
- ad hoc run, plot, metadata, primary NetCDF, and constant-temperature paths
- production root, manifest, annual NetCDF, and year-specific plot paths
- output collision and overwrite checks
- ad hoc `run_info.json`
- production `production_run.json`
- NetCDF writing with unsupported `None` attributes removed
- named-branch git provenance
- tracked runtime dirty-state detection

### `src/plot_diagnostics.py`

Writes calculation diagnostics:

- `fig1_mass_continuity()`
- `fig1_benchmark_mass_continuity()`
- `fig2_mass_advection_residual_timeseries()`
- `fig3_advection_components_timeseries()`
- `fig4_temperature_derivative_timeseries()`
- `fig5_benchmark_comparison()`
- `fig6_benchmark_heating_comparison()`

### `src/plot_results.py`

Writes summary plots:

- `plot_budget_terms_hourly()`
- `plot_budget_terms_day_bin()`
- `plot_constant_T_results()`

The plots normalize by domain volume and use `advection_error` for the current
advection and diabatic uncertainty shading.

### `scripts/run_budget.py`

Is the common calculation entrypoint. It:

- resolves CLI values into the dataclass contracts
- selects ad hoc or production output behavior
- writes or requires a production manifest
- loads and validates the selected input adapter
- resolves the domain
- calls `budget.calculate_budget()`
- writes primary and optional constant-temperature outputs
- writes optional plots

Backend versions may differ in how paths or the Dask client are initialized,
but they must preserve the same canonical input and budget output contracts.

### `scripts/staged_arco_retrieval.py`

Present on staged tips. It:

- accepts the shared domain, pressure, time, and benchmark options
- expands production year ranges into seasonal windows
- optionally splits requests into daily or monthly staging chunks
- avoids a store-wide Dask graph while selecting compact staged data
- materializes selected tile variables before normalizing their Zarr chunks
- retries recognized transient open or write failures
- writes indexed compact Zarr tiles
- skips only an exact tile already registered in the cache

It is a data-acquisition adapter and does not calculate the heat budget.

### `scripts/consolidate_staged_arco_cache.py`

Present on sharded staged tips. This thin backend-neutral CLI calls
`src_arco.consolidation` to validate completed yearly cache shards and publish
the combined `cache.sqlite`. It is cache publication and integrity logic, not
heat-budget logic.

### `scripts/staged_arco_campaign.py`

Present on sharded staged tips. Its `init` action creates or validates immutable
campaign metadata. Its `finalize-year` action validates one yearly cache,
records file hashes and scheduler/Git provenance, and writes the completion
marker last.

### `cache_viz/visualize_staged_arco_cache.py`

Opens an indexed staged cache read-only and writes:

- `tiles.csv`, a sortable inventory of registered tiles and their coverage
- `coverage_timeline.png`, grouped by region, vertical domain, and benchmark
  availability

It reports catalog state and missing tile paths without modifying the cache.

### `archive/scripts/update_run_catalog.py`

Reconstructs a sortable CSV catalog for legacy single-run PBS results by
combining `run_info.json` metadata with matching log-file exit status. It
builds the full result in memory and replaces the output atomically.

### `scripts/closure_testing.py`

This tracked file is currently an empty placeholder on the tips where it is
present. Closure coverage belongs to `tests/test_budget_closure.py`.

### Google Batch Deployment Modules

Present under `deployment/gcp/` on the Google staged tip:

- `campaign.py`
  - validates and normalizes campaign JSON
  - resolves regions, yearly time windows, and task-to-year mappings
- `render_batch_job.py`
  - renders digest-pinned yearly Google Batch job specifications
- `run_yearly_retrieval.py`
  - runs one year's staged retrieval and publishes its shard
- `shard_artifacts.py`
  - builds and validates shard manifests, paths, file hashes, and tile
    signatures
- shell entrypoints
  - build the retrieval image
  - configure Batch resources
  - submit the campaign

### Scheduler And Deployment Adapters

Files under `schedulers/` and `deployment/` may:

- select PBS, Slurm, or Google queueing
- map array tasks to years
- provide backend paths and environment activation
- stage or consolidate ARCO cache artifacts
- submit retrieval and calculation jobs

They must not redefine the scientific terms, signs, canonical dataset, staged
cache schema, or budget output schema.

## 7. Data Contracts

### 7.1 Canonical Loaded Dataset

After `io.standardize_era5_dataset()`, downstream code expects coordinates:

- `time`
- `level`
- `lat`
- `lon`

Required variables:

- `T(time, level, lat, lon)`
- `u(time, level, lat, lon)`
- `v(time, level, lat, lon)`
- `w(time, level, lat, lon)`
- `sp(time, lat, lon)`

Optional surface variables:

- `T2m(time, lat, lon)`
- `u10(time, lat, lon)`
- `v10(time, lat, lon)`

Coordinate conventions:

- `level` is in Pa and strictly decreasing
- `lat` and `lon` are cell centers
- `lat` and `lon` are strictly increasing
- longitude is normalized to `[-180, 180]`
- time spacing is regular

The input adapter is no longer relevant once this contract has been
established.

### 7.2 Domain And Halo Datasets

After `grid.determine_domain()`:

- `ds_domain` is the interior control-volume grid
- `ds_halo` adds one horizontal cell around `ds_domain`
- selected center coordinates remain `lat` and `lon`
- bounds are reconstructed from adjacent centers
- `DomainSpec` contains the actual resolved control-volume boundaries

Both datasets carry:

- `lat_start`, `lat_end`
- `lon_start`, `lon_end`
- `p_start`, `p_end`, `p_mid`
- `lat_cell_id`, `lon_cell_id`, `p_cell_id`

### 7.3 Geometry And Weights

Geometry outputs:

- `A_horizontal(lat, lon)`
- `A_east(level, lat)`
- `A_west(level, lat)`
- `A_south(level, lon)`
- `A_north(level, lon)`
- `V_cell(level, lat, lon)`

Weight outputs:

- `W_top(time, lat, lon)`
- optional `W_bottom(time, lat, lon)`
- `W_east(time, level, lat)`
- `W_west(time, level, lat)`
- `W_south(time, level, lon)`
- `W_north(time, level, lon)`
- `W_volume(time, level, lat, lon)`

### 7.4 Budget Output Dataset

The entrypoint calls `calculate_budget()` with integral diagnostics enabled.
Its primary result contains the following core, geometric, closure, and
advection fields.

Core temperature and volume fields:

- `d_dt_T`
  - centered derivative of the volume integral of `T`
- `dT_dt`
  - `domain_volume * d(T_domain_avg)/dt`
  - used as `S` in the diabatic residual
- `dT_dt_2`
  - `d_dt_T - T_domain_avg * dV_dt`
- `T_domain_avg`
- `T_scale`
- `domain_volume`
  - volume from grid-cell geometry and occupancy weights
- `domain_volume_true`
  - direct horizontal integral of active pressure depth
- `dV_dt`
  - derivative of `domain_volume`
- `dV_dt_true`
  - derivative of `domain_volume_true`

Heat-budget and closure fields:

- `advection_term`
  - temperature-anomaly heat advection, positive into the domain
- `adiabatic_term`
- `diabatic_term`
  - `dT_dt - advection_term - adiabatic_term`
- `net_mass_advection`
  - positive into the domain
- `residual_heat`
  - `(net_mass_advection - dV_dt) * T_domain_avg`
- `advection_error`
  - `(net_mass_advection - dV_dt) * T_scale`
- `abs_mass_advection_residual_fraction`

Per-face fields:

- `flux_contribution_west`
- `flux_contribution_east`
- `flux_contribution_south`
- `flux_contribution_north`
- `flux_contribution_top`
- optional `flux_contribution_bottom`
- `mass_flux_contribution_west`
- `mass_flux_contribution_east`
- `mass_flux_contribution_south`
- `mass_flux_contribution_north`
- `mass_flux_contribution_top`
- optional `mass_flux_contribution_bottom`

All per-face contributions use the positive-into-domain sign convention.
Surface-pressure runs contain 26 primary variables. Fixed-pressure-bottom runs
add the two bottom-face fields for 28.

### 7.5 Optional Benchmark Output

When benchmark variables are available and
`--include-benchmark-variables` is enabled for a
1 hPa-to-surface-pressure domain, the result also contains seven advective
aggregate diagnostic fields:

- `benchmark_mass_flux_net`
- `calculated_mass_flux_net_lateral`
- `benchmark_heat_flux_net`
- `calculated_heat_flux_net_lateral_full`
- `calculated_heat_flux_net_lateral_full_benchmark_mass`
- `calculated_heat_flux_net_lateral`
- `benchmark_heat_flux_net_lateral_prime`

It also contains eight signed per-face benchmark fields:

- `benchmark_mass_flux_north`
- `benchmark_mass_flux_south`
- `benchmark_mass_flux_east`
- `benchmark_mass_flux_west`
- `benchmark_heat_flux_north`
- `benchmark_heat_flux_south`
- `benchmark_heat_flux_east`
- `benchmark_heat_flux_west`

These 15 fields are retained in NetCDF output because downstream diagnostics
need both aggregate and wall-resolved comparisons.

The same optional result contains 12 heating-benchmark fields:

- `benchmark_vithe`
- `benchmark_viec`
- `benchmark_vithed`
- `benchmark_thermal_content`
- `benchmark_storage_term`
- `benchmark_adiabatic_term`
- `benchmark_heat_flux_divergence`
- `benchmark_mass_residual`
- `benchmark_residual_heat`
- `benchmark_diabatic_term_physical`
- `benchmark_diabatic_term`
- `calculated_diabatic_term_physical`

These retain the reduced ERA5 source values, the definition-matched
volume-average comparison, and the secondary physical full-temperature
comparison described in `docs/full-column-benchmarking-strategy.md`.

### 7.6 Ad Hoc Run Metadata

Ad hoc `run_info.json` contains:

```json
{
  "run_id": "...",
  "scheduler": "pbs, slurm, or null",
  "scheduler_job_id": "...",
  "scheduler_array_task_id": "...",
  "pbs_job_id": "...",
  "generated_at": "...",
  "run_root": "...",
  "plot_dir": "...",
  "request": { "...": "..." },
  "source_spec": { "...": "..." },
  "domain_spec": { "...": "..." },
  "surface_behaviour": { "...": "..." },
  "git": {
    "branch": "...",
    "commit": "...",
    "dirty": false
  },
  "cli_args": { "...": "..." }
}
```

The scheduler fields are nullable for manual runs.

### 7.7 Production Metadata And Layout

Production output uses:

```text
<production-output-dir>/
├── production_run.json
├── annual/
│   └── heat_budget_<year>.nc
└── plots/
    └── <year>/
```

`production_run.json` stores campaign-wide provenance. It is distinct from the
ad hoc per-run metadata contract.

## 8. Test Inventory

The repository-level `pytest.ini` collects tests from `tests/` and excludes
generated, archived, notebook, and result directories.

Shared baseline tests:

- `tests/test_benchmark_diagnostics.py`
  - aggregate benchmark formulas
  - per-face benchmark output schema and signs
  - full-column domain guard and analytic heating equations
  - benchmark Figures 1, 5, and 6
- `tests/test_budget_closure.py`
  - exact and analytic mass and heat-advection closure cases
- `tests/test_grid.py`
  - domain cropping, bounds, halo alignment, areas, and volumes
- `tests/test_integrals.py`
  - currently empty
- `tests/test_io.py`
  - ARCO retry behavior and benchmark loading
- `tests/test_run_budget.py`
  - CLI resolution
  - ad hoc and production modes
  - NetCDF output
  - plotting and constant-temperature controls
  - benchmark dispatch
- `tests/test_run_outputs.py`
  - PBS and Slurm run IDs
  - metadata and git provenance
  - production paths and manifests
  - NetCDF collision and serialization behavior
- `tests/test_update_run_catalog.py`
  - archived run-catalog reconstruction and failure safety
- `tests/test_weights.py`
  - horizontal, vertical, and volume weights
  - direct true-volume and volume-tendency diagnostics

Staged-tip addition:

- `tests/test_arco_cache.py`
  - compact wall-only tile construction and reconstruction
  - cache identity and coverage
  - local offline loading
  - time-tile mosaics and gap/conflict rejection
  - benchmark cache enforcement
  - retry and timeout behavior
  - equivalence between staged and full-input budget results
  - explicit rejection of unsupported staged surface variables
- `tests/test_staged_arco_shards.py`
  - immutable campaign initialization and legacy-root protection
  - per-year shard completion and hourly-coverage validation
  - atomic combined-catalog publication and failure preservation
  - canonical-dataset and budget equivalence with the legacy cache layout
  - missing-year and integrity rejection

Google staged-tip addition:

- `tests/test_gcp_batch_deployment.py`
  - campaign normalization and validation
  - digest-pinned yearly Batch rendering
  - shard publication and completion markers
  - cache consolidation, integrity, coverage, and path safety

Validated portable baseline at this document update:

- `81 passed` under the `ehb-dev` environment declared in `environment.yml`
- `python scripts/run_budget.py --help` exits successfully without starting a
  Dask cluster

Isolated cross-tip validation at this document update:

- `drac_development_2_staged`: `120 passed`
- `google_development_staged`: `139 passed` with one Matplotlib date-locator
  warning
- `production_development_staged`: `147 passed`

Branch-specific suites must pass before their corresponding retrieval or
production deployment.

## 9. Shared Versus Branch-Specific Responsibilities

The following are shared scientific contracts:

- positive-into-domain heat and mass advection
- diabatic and mass-closure formulas
- canonical loaded dataset
- domain, geometry, and weight definitions
- staged cache schema and offline reconstruction on staged tips
- budget output variable names and meanings
- benchmark output signs and fields

The following may vary by branch tip:

- PBS, Slurm, or Google queueing
- retrieval-job construction
- deployment resources and container execution
- configured regions
- machine paths and environment activation
- cache transfer and consolidation

Backend code must terminate at the data-source or execution boundary. It must
not fork the scientific implementation or reinterpret its signs and outputs.

## 10. Current Implementation Notes

- The main calculation entrypoint is `scripts/run_budget.py`.
- `src/budget.py` orchestrates scientific modules after canonical data loading.
- Plotting modules live under `src/`.
- Legacy code inside inactive comments or string blocks is not part of the
  active API.
- The primary budget output is intentionally broader than the plotted fields
  because wall, closure, true-volume, and benchmark variables are required by
  downstream analysis.
- A real-data workflow is validated only when its scheduler job, commit,
  configuration, output path, and scientific checks are recorded.
