# Eulerian Heat Budget - Current Code Outline

May 4, 2026

## 1. Purpose

This document describes the code as it exists today in this repository. It is a code-faithful reference for the current Eulerian heat-budget workflow in pressure coordinates, including the runtime pipeline, module boundaries, data contracts, outputs, associated scripts, and the present shape of the test suite.

The implementation computes volume-integrated temperature storage, advective fluxes, adiabatic heating, and a residual diabatic term over a regional control volume. It also supports ad hoc diagnostic runs, optional constant-temperature validation runs, and production-style yearly NetCDF output.

Only active code paths are documented here. Legacy code inside commented or triple-quoted blocks is not treated as current behavior.

## 2. Runtime Pipeline

The current end-to-end budget flow is:

1. `src/cli.py` parses command-line inputs.
2. `scripts/run_budget.py` combines CLI values with defaults from `src/config.py` and constructs:
   - `specs.DomainRequest`
   - `specs.SurfaceBehaviour`
   - `specs.DataSourceConfig`
   - runtime controls for diagnostic plots and constant-temperature testing
   - optional production output settings
3. `src/run_outputs.py` resolves git provenance for runtime source paths under `src`, `scripts`, and `schedulers`.
4. If `--init-production-manifest` is used, the run writes a production manifest and exits before loading data.
5. For normal ad hoc runs, `run_outputs.prepare_run_paths()` creates a run directory under the configured plot output root.
6. For yearly production runs, `run_outputs.prepare_production_paths()` resolves:
   - shared production manifest path
   - annual NetCDF output path
   - year-specific plot directory
7. `src/io.py` loads ERA5 data from local NetCDF files or ARCO Zarr and standardizes it to the internal schema.
8. `src/validate.py` enforces the canonical dataset schema.
9. When using `arco_era5`, `src/io.py::load_arco_benchmark_fluxes()` also loads benchmark vertically integrated heat and mass fluxes for diagnostic comparison.
10. `src/grid.py::determine_domain()` crops the domain and returns:
    - `ds_domain`
    - `ds_halo`
    - `DomainSpec`
11. Ad hoc runs write `run_info.json` through `src/run_outputs.py::write_run_info()`.
12. `src/budget.py::calculate_budget()` builds geometry and weights, computes storage/advection/adiabatic/diabatic terms, merges advection diagnostics, and optionally writes diagnostic figures.
13. Production yearly runs serialize the resulting budget dataset to `annual/heat_budget_<year>.nc`.
14. If diagnostic plots are enabled, `src/plot_results.py` writes raw, 24-hour-smoothed, and daily summary plots.
15. If the constant-temperature test is enabled, `scripts/run_budget.py` runs a second budget calculation with `T` replaced by `T_scale` and writes comparison plots when plotting is enabled.

Under `__main__`, `scripts/run_budget.py` starts a Dask distributed `Client` with four workers, one thread per worker, processes enabled, and an 8 GB memory limit per worker.

## 3. Repository Structure

```text
eulerian_heat_budget_6hr_test/
|-- docs/
|   |-- Eulerian Heat Budget - Reformulation.md
|   `-- code_outline.md
|-- logs/
|-- results/
|-- schedulers/
|   |-- schedule_check_pbl.sh
|   |-- schedule_run_budget.sh
|   `-- schedule_run_budget_production.sh
|-- scripts/
|   |-- check_pbl.py
|   `-- run_budget.py
|-- src/
|   |-- __init__.py
|   |-- budget.py
|   |-- cli.py
|   |-- config.py
|   |-- grid.py
|   |-- integrals.py
|   |-- io.py
|   |-- plot_diagnostics.py
|   |-- plot_results.py
|   |-- run_outputs.py
|   |-- specs.py
|   |-- terms.py
|   |-- time_utils.py
|   |-- validate.py
|   `-- weights.py
`-- tests/
    |-- test_budget_closure.py
    |-- test_check_pbl.py
    |-- test_grid.py
    |-- test_integrals.py
    |-- test_io.py
    |-- test_plot_results.py
    |-- test_run_budget.py
    |-- test_run_outputs.py
    |-- test_time_utils.py
    `-- test_weights.py
```

## 4. Module Reference

### `src/config.py`

Defines project constants and runtime defaults:

- Physical constants: `g`, `R_value`, `R_earth`, `cp`
- Default data-source settings:
  - `DEFAULT_DATA_SOURCE`
  - `DEFAULT_LOCAL_PATH`
  - `DEFAULT_ARCO_PATH`
  - `DEFAULT_ARCO_TOKEN`
  - `DEFAULT_ARCO_OPEN_MAX_ATTEMPTS`
  - `DEFAULT_ARCO_OPEN_RETRY_BASE_DELAY_SECONDS`
  - `DEFAULT_TIME_START`
  - `DEFAULT_TIME_END`
- Default domain settings:
  - `DEFAULT_BBOX`
  - `DEFAULT_MARGIN_N`
  - `DEFAULT_ZG_TOP_PA`
  - `DEFAULT_ZG_BOT_MODE`
  - `DEFAULT_ZG_BOT_PA`
- Surface-behaviour defaults:
  - `DEFAULT_ALLOW_BOTTOM_OVERFLOW`
  - `DEFAULT_USE_SURFACE_VARIABLES`
  - `DEFAULT_SURFACE_VARIABLE_MODE`
- Output, runtime, and Dask defaults:
  - `DEFAULT_PLOTS_OUTPUT`
  - `DEFAULT_DIAGNOSTIC_PLOTS`
  - `DEFAULT_CONSTANT_TEMPERATURE_TEST`
  - `DEFAULT_CHUNKS_3D1`
  - chunk-size helpers `n_time`, `n_lat`, `n_lon`

`config.py` does not perform calculations; it is the central source of default values used by scripts and CLI helpers.

### `src/specs.py`

Defines the dataclass-based contracts used throughout the run:

- `DataSourceConfig`
  - selects `local_era5` or `arco_era5`
  - stores source paths, ARCO token, chunking, and time window
- `DomainRequest`
  - stores requested bounding box, margin, top pressure, and bottom-boundary mode
- `SurfaceBehaviour`
  - stores bottom-overflow behavior
  - toggles optional surface-variable use
  - stores `surface_variable_mode`
- `DomainSpec`
  - stores resolved lat/lon bounds actually used after cropping
  - stores resolved top and bottom control-volume pressures
  - validates bottom-boundary consistency

This module is the canonical definition of what the domain request, resolved domain, data source, and surface treatment mean in the current codebase.

### `src/cli.py`

Builds the command-line parser. It currently parses:

- Horizontal bounds: `--lat-min`, `--lat-max`, `--lon-min`, `--lon-max`
- Horizontal margin: `--margin-n`
- Vertical control-volume settings:
  - `--zg-top-pa`
  - `--zg-bottom`
  - `--zg-bottom-pa`
- Surface behavior:
  - `--allow-bottom-overflow` / `--no-allow-bottom-overflow`
  - `--use-surface-variables` / `--no-use-surface-variables`
  - `--surface-variable-mode`
- Data source and time window:
  - `--data-source`
  - `--time-start`
  - `--time-end`
- Runtime toggles:
  - `--diagnostic-plots` / `--no-diagnostic-plots`
  - `--constant-temperature-test` / `--no-constant-temperature-test`
- Production-output controls:
  - `--production-output-dir`
  - `--init-production-manifest`
  - `--production-start-year`
  - `--production-end-year`
  - `--overwrite-output`

Most parser defaults are `None` so `scripts/run_budget.py` can distinguish unspecified values from explicit user choices.

### `src/io.py`

Loads and standardizes input data. Current responsibilities are:

- `load_dataset()`
  - dispatches to `_load_local_era5()` or `_load_arco_era5()`
  - runs `standardize_era5_dataset()` before returning
- Local ERA5 load path:
  - loads `T`, `u`, `v`, `w`, and `sp`
  - optionally loads `T2m`, `u10`, and `v10`
  - merges component datasets
- ARCO ERA5 load path:
  - opens the ARCO Zarr store through `_open_arco_zarr_with_retry()`
  - renames ERA5 variable names to the internal schema
  - optionally includes surface variables
- `standardize_era5_dataset()`
  - renames common external coordinate names to `time`, `level`, `lat`, `lon`
  - drops auxiliary coordinates such as `number`, `expver`, `step`, and `surface`
  - enforces canonical dimension ordering
  - converts pressure levels to Pa when needed
  - converts temperature units to Kelvin when needed
  - normalizes longitudes into `[-180, 180]`
  - applies the requested time slice
  - chunks the dataset for Dask workflows
- `load_arco_benchmark_fluxes()`
  - loads vertically integrated ARCO benchmark flux variables
  - standardizes time/lat/lon coordinates and longitude convention
  - chunks the result for later diagnostic comparison
- `_open_arco_zarr_with_retry()`
  - retries transient DNS, connection, server-disconnect, timeout, and service-unavailable failures using exponential backoff

The current internal variable names expected downstream are:

- Required 4D fields: `T`, `u`, `v`, `w`
- Required 3D field: `sp`
- Optional surface fields: `T2m`, `u10`, `v10`

### `src/validate.py`

Performs strict validation of the standardized dataset:

- required dims exist: `time`, `level`, `lat`, `lon`
- required 4D variables use dims `("time", "level", "lat", "lon")`
- required 3D variable `sp` uses dims `("time", "lat", "lon")`
- each coordinate is 1D over itself
- levels are strictly decreasing
- lat/lon are strictly increasing
- temperature units are Kelvin
- time spacing is regular when multiple time steps are present

This module raises errors instead of coercing bad inputs, to prevent silent scientific mistakes.

### `src/grid.py`

Provides domain resolution and geometric operators.

Current domain logic:

- `determine_domain(ds, request, eager_loading=False)` interprets input `lat` and `lon` as cell-center coordinates.
- The requested bbox selects grid cells whose centers lie inside the requested bounds.
- `margin_n` is applied in cell space and must currently be at least `1`, because `ds_halo` is built from `margin_n - 1`.
- The function returns:
  - `ds_domain`: the interior control-volume grid
  - `ds_halo`: the same domain with a one-cell horizontal pad, used for wall-flux and wall-weight calculations
  - `DomainSpec`: resolved bounds and vertical boundary settings
- The returned datasets carry cell-center coordinates plus bound metadata:
  - `lat_start`, `lat_end`
  - `lon_start`, `lon_end`
  - `p_start`, `p_end`, `p_mid`
  - `lat_cell_id`, `lon_cell_id`, `p_cell_id`

Current geometry and alignment helpers:

- `crop_to_target_grid(ds, target)`
  - nearest-neighbour aligns a dataset to coordinates shared with a target dataset
- `get_boundary_line_elements(ds)`
  - returns wall line elements `dl_east`, `dl_west`, `dl_south`, and `dl_north` in meters
- `get_horizontal_cell_areas(ds)`
  - returns top-face horizontal area `A_horizontal(lat, lon)` in `m2`
- `get_vertical_cell_areas(ds)`
  - returns wall areas:
    - `A_east(level, lat)`
    - `A_west(level, lat)`
    - `A_south(level, lon)`
    - `A_north(level, lon)`
  - units are `m*Pa`
- `get_cell_volumes(ds)`
  - returns `V_cell(level, lat, lon)` in `m2*Pa`

The grid module handles geometry, domain bookkeeping, and benchmark-grid alignment. It does not compute heat-budget physics.

### `src/weights.py`

Builds occupancy weights for the control volume.

Current horizontal-face weights:

- `area_weights_horizontal(ds_domain, domain_spec)`
  - always returns `W_top(time, lat, lon)`
  - returns `W_bottom(time, lat, lon)` only when the bottom boundary is a fixed pressure level
  - top and bottom weights are binary in the current implementation

Current vertical-wall weights:

- `area_weights_vertical(ds_halo, domain_spec, surface_spec)`
  - returns:
    - `W_east(time, level, lat)`
    - `W_west(time, level, lat)`
    - `W_south(time, level, lon)`
    - `W_north(time, level, lon)`
  - computes wall occupancy from halo-adjacent surface-pressure slices
  - supports both bottom-boundary modes:
    - `surface_pressure`
    - `pressure_level`
  - when `allow_bottom_overflow=True` and the bottom boundary follows surface pressure, bottom-layer wall weights may exceed `1`

Current volume weights:

- `volume_weights(ds_domain, domain_spec, surface_spec)`
  - returns `W_volume(time, level, lat, lon)`
  - computes overlap of each pressure layer with the active control-volume column
  - supports the same bottom-boundary logic as the wall weights
  - when `allow_bottom_overflow=True` and the bottom boundary follows surface pressure, the bottom-layer weight may exceed `1`

### `src/integrals.py`

Contains pure integration operators:

- `area_integral(field_2d, area_2d, weights_2d)`
- `volume_integral_pcoords(field_3d, volume_3d, weights_3d)`

These functions multiply the field by geometric factors and weights, then sum over all non-time dimensions. They validate that area integrals reduce two non-time dimensions and volume integrals reduce three non-time dimensions.

### `src/time_utils.py`

Contains utilities for regular time axes:

- `infer_timestep_seconds()`
  - infers the first timestep in seconds
- `require_regular_time()`
  - validates constant spacing and returns the timestep in seconds
- `samples_for_duration()`
  - converts a duration in seconds into an integer sample count
- `integrate_rate_over_time()`
  - multiplies a rate time series by the dataset timestep

`validate.py` uses this module to reject irregular time axes. `plot_results.py` and `plot_diagnostics.py` use it for cadence-aware smoothing and time integration.

### `src/terms.py`

Computes the physical terms and associated helpers.

Current public helpers:

- `compute_domain_volume()`
- `compute_time_derivative()`
- `compute_storage()`
- `prepare_advective_faces()`
- `compute_advective_term()`
- `compute_adiabatic_term()`
- `compute_diabatic_term()`
- `compute_T_domain_average()`
- `compute_advective_benchmark_fluxes()`

Important internal helpers:

- `_adjust_surface_field()`
  - blends model-level and surface variables in the surface-adjacent layer when `use_surface_variables=True`
- `_trim_advective_state()`
  - reduces domain and halo datasets to only the variables needed for advection
- `_drop_face_coords()`
  - removes scalar/bound coordinates that no longer apply after extracting a face

Current advection workflow:

1. `budget.calculate_budget()` constructs anomaly-temperature advection inputs by subtracting `T_domain_avg` unless the constant-temperature diagnostic path is active.
2. `prepare_advective_faces()` trims domain and halo inputs to the needed advection variables.
3. If surface variables are enabled, `prepare_advective_faces()` adjusts `u`, `v`, and `T` near the surface using `u10`, `v10`, and `T2m`.
4. It constructs face-centered quantities on west, east, south, north, and top faces, and bottom when the bottom boundary is fixed pressure.
5. `compute_advective_term()` integrates signed heat and optional mass face fluxes using geometric areas and area weights.

Current outputs from `compute_advective_term()`:

- always:
  - `net_heat_advection`
  - per-face `flux_contribution_*`
- when `integral_diagnostics_flag=True`:
  - `net_mass_advection`
  - per-face `mass_flux_contribution_*`
  - `abs_mass_advection_residual_fraction`

Current thermodynamic terms:

- `compute_storage()` computes the centered time derivative of volume-integrated temperature.
- `compute_adiabatic_term()` integrates `w * R * T / (cp * p)`.
- `compute_diabatic_term()` currently uses the code sign convention `D = S + A - C`.
- `compute_advective_benchmark_fluxes()` converts ARCO vertically integrated benchmark mass and heat fluxes onto the lateral control-volume walls for diagnostic comparison.

### `src/budget.py`

Orchestrates the full budget calculation.

Current `calculate_budget()` workflow:

1. Builds horizontal areas, vertical areas, and cell volumes.
2. Builds horizontal, vertical, and volume weights.
3. Computes:
   - storage term `d_dt_T`
   - domain volume `domain_volume`
   - volume tendency `dV_dt`
   - domain-mean temperature `T_domain_avg`
   - normalized temperature tendency diagnostics `dT_dt` and `dT_dt_2`
4. Prepares anomaly or constant-temperature advection inputs.
5. Runs `prepare_advective_faces()` and `compute_advective_term()`.
6. Estimates advection uncertainty as:
   `advection_error = (dV_dt + net_mass_advection) * T_scale`
7. Computes:
   - `adiabatic_term`
   - `diabatic_term`
8. Merges the scalar budget terms with advection diagnostics.
9. If benchmark fluxes are provided, aligns them to the halo grid and computes lateral benchmark comparison terms.
10. If plotting is enabled, writes diagnostic figures through `plot_diagnostics.py`.

The active runtime calls `calculate_budget()` with `integral_diagnostics_flag=True`, so the returned dataset includes both heat-budget terms and mass/advection diagnostics.

Current main output fields are:

- `d_dt_T`
- `dT_dt`
- `dT_dt_2`
- `dV_dt`
- `advection_term`
- `advection_error`
- `adiabatic_term`
- `diabatic_term`
- `T_domain_avg`
- `domain_volume`
- `T_scale`
- `net_mass_advection`
- per-face `flux_contribution_*`
- per-face `mass_flux_contribution_*`
- `abs_mass_advection_residual_fraction`

### `src/run_outputs.py`

Handles run metadata and output-path setup.

Current dataclasses:

- `RunPaths`
- `GitProvenance`
- `ProductionPaths`

Current ad hoc responsibilities:

- `resolve_run_id()`
  - uses `PBS_JOBID` when available
  - otherwise generates a manual timestamp-and-pid run id
- `prepare_run_paths()`
  - creates the run root and plot directory
  - returns `RunPaths`
- `write_run_info()`
  - serializes run metadata into `run_info.json`

Current production responsibilities:

- `prepare_production_paths()`
  - creates shared `annual/` and `plots/` directories
  - resolves `production_run.json`
  - resolves `annual/heat_budget_<year>.nc` and `plots/<year>/` for yearly runs
- `write_production_manifest()`
  - writes shared campaign metadata and exits before data loading in manifest-init mode
- `require_production_manifest()`
  - fails yearly production runs if the shared manifest is absent
- `resolve_production_year()`
  - derives the year from `time_start` and `time_end`
  - rejects cross-year production slices
- `require_output_path()`
  - prevents accidental overwrite unless `--overwrite-output` is used
- `write_budget_result()`
  - drops `None` attributes and writes the budget dataset to NetCDF
- `resolve_git_provenance()`
  - captures current branch, commit, and dirty state
  - dirty-state checks are limited to tracked runtime sources under `src`, `scripts`, and `schedulers`
  - ignores generated noise such as `__pycache__` and `.pyc`

### `src/plot_diagnostics.py`

Produces diagnostic figures written by `budget.calculate_budget()` when plotting is enabled.

Current figure functions:

- `fig1_mass_continuity()`
  - mass-continuity scatter and binned residual plots
- `fig2_mass_advection_residual_timeseries()`
  - mass residual, cumulative mass terms, relative mass error, and component time series
- `fig3_advection_components_timeseries()`
  - heat-advection component diagnostics and advection-error comparison
- `fig4_temperature_derivative_timeseries()`
  - storage and domain-average temperature tendency comparison
- `fig5_benchmark_comparison()`
  - lateral mass and heat flux comparison against ARCO benchmark fluxes

These focus on mass continuity, flux decomposition, accumulated residuals, temperature-tendency diagnostics, and optional benchmark-flux comparison.

### `src/plot_results.py`

Produces summary budget figures after a run completes.

Current figure functions:

- `plot_budget_terms_timeseries()`
  - per-timestep view with optional elapsed-time smoothing
  - smoothing windows are derived from the dataset timestep
- `plot_budget_terms_daily()`
  - daily-binned and daily-integrated view using the dataset's actual timestep
- `plot_constant_T_results()`
  - compares the advection-error estimate against the constant-`T` diagnostic rerun

The current plotting code uses normalized quantities based on `domain_volume`, and it shades advection-related uncertainty using `advection_error` when available.

### `scripts/run_budget.py`

This is the current executable entrypoint.

Current responsibilities:

- inserts the project root into `sys.path`
- parses CLI arguments
- builds `DomainRequest`, `SurfaceBehaviour`, `DataSourceConfig`, runtime controls, and optional production options
- resolves git provenance
- initializes a production manifest and exits when requested
- prepares ad hoc run paths or production yearly output paths
- loads and validates the source dataset
- loads ARCO benchmark fluxes when using `arco_era5`
- resolves `ds_domain`, `ds_halo`, and `DomainSpec`
- writes ad hoc `run_info.json`
- calls `budget.calculate_budget()`
- writes production yearly NetCDF output when production mode is active
- writes summary plots when diagnostic plotting is enabled
- optionally runs the second constant-`T` diagnostic budget calculation

Current runtime behavior under `__main__`:

- starts a Dask distributed `Client`
- uses hard-coded worker settings:
  - `n_workers=4`
  - `threads_per_worker=1`
  - `processes=True`
  - `memory_limit="8GB"`

### `scripts/check_pbl.py`

This utility checks planetary boundary layer height from ARCO ERA5 for a given year and bounding box. It is used to inform the choice of `DEFAULT_ZG_TOP_PA`.

Current behavior:

- opens ARCO ERA5 with retry logic matching the budget loader
- selects `boundary_layer_height` and `geopotential`
- limits analysis to June, July, and August for the requested year
- estimates pressure at the local PBL top using geopotential height and log-pressure interpolation
- summarizes PBL height and PBL-top pressure statistics
- writes year-specific `run_info.json`
- writes spatial plots for minimum, p01, and p05 PBL-top pressure

### `schedulers/`

The repository includes scheduler wrappers for batch environments:

- `schedule_run_budget.sh`
  - launches the standard budget workflow
- `schedule_run_budget_production.sh`
  - launches production-style budget runs
- `schedule_check_pbl.sh`
  - launches the PBL checking utility

The current wrappers resolve the repository root from `PROJECT_ROOT`, `PBS_O_WORKDIR`, submit directory, or script location, and allow `LOG_DIR` to be overridden through the environment.

## 5. Data Contracts

### 5.1 Canonical Loaded Dataset

After `io.standardize_era5_dataset()`, downstream code expects:

- coordinates:
  - `time`
  - `level`
  - `lat`
  - `lon`
- required variables:
  - `T(time, level, lat, lon)`
  - `u(time, level, lat, lon)`
  - `v(time, level, lat, lon)`
  - `w(time, level, lat, lon)`
  - `sp(time, lat, lon)`
- optional variables:
  - `T2m(time, lat, lon)`
  - `u10(time, lat, lon)`
  - `v10(time, lat, lon)`

Required coordinate conventions at this stage:

- `level` in Pa
- `level` strictly decreasing
- `lat` strictly increasing
- `lon` strictly increasing
- longitudes normalized to `[-180, 180]`
- time coordinate regular when multiple time steps are present

### 5.2 ARCO Benchmark Flux Dataset

When the source is `arco_era5`, benchmark flux loading maps ARCO variables to:

- `Fx_heat`
- `Fy_heat`
- `Fx_mass`
- `Fy_mass`

The benchmark dataset is standardized to `time`, `lat`, and `lon`, normalized to `[-180, 180]`, chunked, then later cropped to the halo grid for lateral wall comparison.

### 5.3 Domain And Halo Datasets

After `grid.determine_domain()`:

- input horizontal coordinates are treated as cell centers
- `ds_domain` is the interior control-volume grid used for volume integrals and top-face quantities
- `ds_halo` carries a one-cell horizontal pad used for wall-face quantities
- both datasets carry bound metadata:
  - `lat_start`, `lat_end`
  - `lon_start`, `lon_end`
  - `p_start`, `p_end`, `p_mid`
  - `lat_cell_id`, `lon_cell_id`, `p_cell_id`

`DomainSpec` stores the resolved control-volume bounds actually used after center selection and margin application.

### 5.4 Geometry And Weight Outputs

Current geometry outputs:

- `A_horizontal(lat, lon)`
- `A_east(level, lat)`
- `A_west(level, lat)`
- `A_south(level, lon)`
- `A_north(level, lon)`
- `V_cell(level, lat, lon)`
- `dl_east(lat)`
- `dl_west(lat)`
- `dl_south(lon)`
- `dl_north(lon)`

Current weight outputs:

- `W_top(time, lat, lon)`
- optional `W_bottom(time, lat, lon)`
- `W_east(time, level, lat)`
- `W_west(time, level, lat)`
- `W_south(time, level, lon)`
- `W_north(time, level, lon)`
- `W_volume(time, level, lat, lon)`

### 5.5 Budget Output Dataset

The active runtime calls `budget.calculate_budget()` with integral diagnostics enabled. The returned `xarray.Dataset` contains:

- primary budget terms:
  - `d_dt_T`
  - `dT_dt`
  - `dT_dt_2`
  - `dV_dt`
  - `advection_term`
  - `advection_error`
  - `adiabatic_term`
  - `diabatic_term`
  - `T_domain_avg`
  - `domain_volume`
  - `T_scale`
- advection diagnostics:
  - `net_mass_advection`
  - per-face `flux_contribution_*`
  - per-face `mass_flux_contribution_*`
  - `abs_mass_advection_residual_fraction`

This output is the main input to result plotting and production NetCDF serialization.

### 5.6 Run Metadata And Production Outputs

Ad hoc runs write `run_info.json` with the current structure:

```json
{
  "run_id": "...",
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

Production manifest initialization writes `production_run.json` with campaign-level metadata:

- generation timestamp and optional `PBS_JOBID`
- production start and end year
- shared root, annual output, plot root, and manifest paths
- request, source, surface behavior, git provenance, and CLI args

Production yearly runs require the manifest to exist and write:

- `annual/heat_budget_<year>.nc`
- plots under `plots/<year>/` when diagnostic plotting is enabled

## 6. Testing

Current test files:

- `tests/test_grid.py`
  - covers domain cropping, centered-coordinate bounds metadata, horizontal areas, vertical areas, cell volumes, and halo/core alignment
- `tests/test_weights.py`
  - covers horizontal-face weights, vertical-wall weights, volume weights, and bottom-overflow behavior
- `tests/test_budget_closure.py`
  - covers mass and heat-advection closure behavior for idealized flows
- `tests/test_run_outputs.py`
  - covers ad hoc paths, production paths, production manifests, yearly output guards, NetCDF writing, metadata serialization, and git provenance behavior
- `tests/test_run_budget.py`
  - covers runtime CLI flags, production option validation, default run behavior, plotting toggles, constant-temperature reruns, production manifest initialization, and yearly production output flow
- `tests/test_io.py`
  - covers ARCO retry behavior, benchmark flux loading, and ARCO dataset loading retry dispatch
- `tests/test_time_utils.py`
  - covers timestep inference, duration-to-sample conversion, irregular-time rejection, and timestep-aware rate integration
- `tests/test_plot_results.py`
  - covers smoothing-window selection and cadence-neutral daily totals
- `tests/test_check_pbl.py`
  - covers PBL pressure interpolation, chunk-invariant PBL pressure summaries, spatial pressure metric plots, check-PBL run metadata, and parser behavior
- `tests/test_integrals.py`
  - exists in the test tree but currently has no test definitions

No pytest run is implied by this document. Treat this section as a map of current test intent and coverage, not as a fresh claim that the suite passes.

## 7. Current Implementation Notes

- The document describes current behavior, even where the implementation is transitional.
- Active `grid.determine_domain()` treats horizontal input coordinates as cell centers.
- Commented-out or triple-quoted legacy code in modules such as `grid.py`, `terms.py`, and `io.py` is not part of the active API.
- The plotting modules live under `src/`, not under `scripts/`.
- The main runtime entrypoint is `scripts/run_budget.py`, with `src/budget.py` acting as the orchestration layer used by that script.
- `scripts/check_pbl.py` is a related diagnostic utility, not part of the main budget calculation.
