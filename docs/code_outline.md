# Eulerian Heat Budget - Current Code Outline

March 31st, 2026

## 1. Purpose

This document describes the code as it exists today in this repository. It is a code-faithful reference for the current Eulerian heat-budget workflow in pressure coordinates, including the runtime pipeline, module boundaries, data contracts, outputs, and the present state of the test suite.

The implementation currently computes volume-integrated temperature storage, advective fluxes, and adiabatic heating over a regional control volume, then diagnoses a residual diabatic term from those pieces.

## 2. Runtime Pipeline

The current end-to-end flow is:

1. `src/cli.py` parses command-line inputs.
2. `scripts/run_budget.py` combines CLI values with defaults from `src/config.py` and constructs:
   - `specs.DomainRequest`
   - `specs.SurfaceBehaviour`
   - `specs.DataSourceConfig`
3. `src/run_outputs.py` creates a run directory and resolves git provenance for the run metadata.
4. `src/io.py` loads ERA5 data from either local files or ARCO Zarr and standardizes it to the internal schema.
5. `src/validate.py` enforces the canonical dataset schema.
6. `src/grid.py::determine_domain()` crops the domain and returns:
   - `ds_domain`
   - `ds_halo`
   - `DomainSpec`
7. `src/grid.py` builds geometric areas and cell volumes.
8. `src/weights.py` builds face and volume occupancy weights using the resolved control-volume boundaries and surface-pressure logic.
9. `src/terms.py` prepares face-centered advection inputs and computes storage, advection, adiabatic, and residual diabatic terms.
10. `src/budget.py::calculate_budget()` assembles the final budget dataset and optional diagnostics.
11. `src/plot_diagnostics.py` and `src/plot_results.py` write diagnostic and summary figures.
12. `src/run_outputs.py::write_run_info()` stores a `run_info.json` record alongside the plots.

## 3. Repository Structure

```text
eulerian_heat_budget/
├── docs/
│   └── code_outline.md
├── logs/
├── results/
├── schedulers/
│   └── schedule_run_budget.sh
├── scripts/
│   └── run_budget.py
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
└── tests/
    ├── test_budget_closure.py
    ├── test_grid.py
    ├── test_integrals.py
    ├── test_run_outputs.py
    └── test_weights.py
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
- Output and Dask defaults:
  - `DEFAULT_PLOTS_OUTPUT`
  - `DEFAULT_CHUNKS_3D1`
  - chunk-size helpers `n_time`, `n_lat`, `n_lon`

`config.py` does not perform calculations; it is the central source of default values used by the entrypoint.

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
  - includes validation for bottom-boundary consistency

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

The parser mostly returns `None` defaults. `scripts/run_budget.py` is responsible for filling unspecified values from `config.py`.

### `src/io.py`

Loads and standardizes input data. Current responsibilities are:

- `load_dataset()`
  - dispatches to `_load_local_era5()` or `_load_arco_era5()`
- Local ERA5 load path:
  - loads `T`, `u`, `v`, `w`, and `sp`
  - optionally loads `T2m`, `u10`, and `v10`
  - merges component datasets
- ARCO ERA5 load path:
  - opens the ARCO Zarr store
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

- `determine_domain(ds, request, eager_loading=False)` interprets input `lat` and `lon` as cell-start coordinates, not cell centers.
- The requested bbox is snapped to whole cells.
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

Current geometry helpers:

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

The grid module currently handles only geometry and domain bookkeeping. It does not compute physics.

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

Contains the current pure integration operators only:

- `area_integral(field_2d, area_2d, weights_2d)`
- `volume_integral_pcoords(field_3d, volume_3d, weights_3d)`

These functions operate on xarray arrays and sum over all non-time dimensions after multiplying the field by geometric factors and weights.

Time differencing does not live here in the current code. It is implemented in `src/terms.py`.

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

Important internal helper:

- `_adjust_surface_field()`
  - blends model-level and surface variables in the surface-adjacent layer when `use_surface_variables=True`

Current advection workflow:

1. `prepare_advective_faces()` reduces the datasets to the variables needed for advection.
2. In the standard budget path, `budget.calculate_budget()` first converts advection temperature inputs into anomalies relative to `T_domain_avg`.
3. If surface variables are enabled, `prepare_advective_faces()` adjusts `u`, `v`, and `T` near the surface using `u10`, `v10`, and `T2m`.
4. It constructs face-centered quantities on west, east, south, north, and top faces, and bottom when the bottom boundary is fixed pressure.
5. `compute_advective_term()` integrates the signed face fluxes using geometric areas and area weights.

Current outputs from `compute_advective_term()`:

- always:
  - `net_heat_advection`
  - per-face `flux_contribution_*`
- when `integral_diagnostics_flag=True`:
  - `net_mass_advection`
  - per-face `mass_flux_contribution_*`
  - `abs_mass_advection_residual_fraction`

Current thermodynamic terms:

- `compute_storage()` computes the centered time derivative of volume-integrated temperature
- `compute_adiabatic_term()` integrates `w * R * T / (cp * p)`
- `compute_diabatic_term()` currently uses the code sign convention:
  `D = S + A - C`

### `src/budget.py`

Orchestrates the full budget calculation.

Current `calculate_budget()` workflow:

1. Builds geometric areas and cell volumes.
2. Builds horizontal, vertical, and volume weights.
3. Computes:
   - storage term `d_dt_T`
   - domain volume `domain_volume`
   - volume tendency `dV_dt`
   - domain-mean temperature `T_domain_avg`
   - normalized temperature tendency diagnostics `dT_dt` and `dT_dt_2`
4. Prepares advection inputs.
5. Runs `prepare_advective_faces()` and `compute_advective_term()`.
6. Estimates advection uncertainty as:
   `advection_error = (dV_dt + net_mass_advection) * T_scale`
7. Computes:
   - `adiabatic_term`
   - `diabatic_term`
8. Assembles the output dataset.
9. Optionally writes diagnostic figures through `plot_diagnostics.py`.

Current output dataset fields are:

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

The current implementation also supports a constant-`T` diagnostic rerun through the `test_constant_T` argument.

### `src/run_outputs.py`

Handles run metadata and output-path setup.

Current responsibilities:

- `resolve_run_id()`
  - uses `PBS_JOBID` when available
  - otherwise generates a manual timestamp-and-pid run id
- `prepare_run_paths()`
  - creates the run root and plot directory
  - returns `RunPaths`
- `resolve_git_provenance()`
  - captures current branch, commit, and dirty state
  - dirty-state checks are limited to tracked runtime sources under `src`, `scripts`, and `schedulers`
  - ignores generated noise such as `__pycache__` and `.pyc`
- `write_run_info()`
  - serializes run metadata into `run_info.json`

Current run metadata includes:

- run id and path information
- resolved request and domain specs
- source config
- surface behavior
- git provenance
- raw CLI args
- `PBS_JOBID` when available

### `src/plot_diagnostics.py`

Produces diagnostic figures written by `budget.calculate_budget()` when plotting is enabled.

Current figure functions:

- `fig1_mass_continuity()`
- `fig2_mass_advection_residual_timeseries()`
- `fig3_advection_components_timeseries()`
- `fig4_temperature_derivative_timeseries()`

These focus on mass continuity, flux decomposition, accumulated residuals, and the relationship between different temperature-tendency diagnostics.

### `src/plot_results.py`

Produces summary budget figures after a run completes.

Current figure functions:

- `plot_budget_terms_hourly()`
  - hourly view with configurable rolling smoothing
- `plot_budget_terms_day_bin()`
  - daily-binned and daily-accumulated view
- `plot_constant_T_results()`
  - compares the advection-error estimate against the constant-`T` diagnostic rerun

The current plotting code uses normalized quantities based on `domain_volume`, and it shades advection-related uncertainty using `advection_error` when available.

### `scripts/run_budget.py`

This is the current executable entrypoint.

Current responsibilities:

- inserts the project root into `sys.path`
- parses CLI arguments
- builds `DomainRequest`, `SurfaceBehaviour`, and `DataSourceConfig`
- prepares run directories and git provenance
- loads and validates the source dataset
- resolves `ds_domain`, `ds_halo`, and `DomainSpec`
- writes `run_info.json`
- calls `budget.calculate_budget()`
- writes summary plots
- runs a second constant-`T` diagnostic budget calculation and plots that comparison

Current runtime behavior under `__main__`:

- starts a Dask distributed `Client`
- uses hard-coded worker settings:
  - `n_workers=4`
  - `threads_per_worker=1`
  - `processes=True`
  - `memory_limit="8GB"`

### `schedulers/schedule_run_budget.sh`

This repository also includes a scheduler wrapper for launching the budget workflow in batch environments.

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

### 5.2 Domain And Halo Datasets

After `grid.determine_domain()`:

- input horizontal coordinates have been reinterpreted from cell starts to cell centers
- `ds_domain` is the interior control-volume grid used for volume integrals and top-face quantities
- `ds_halo` carries a one-cell horizontal pad used for wall-face quantities
- both datasets carry bound metadata:
  - `lat_start`, `lat_end`
  - `lon_start`, `lon_end`
  - `p_start`, `p_end`, `p_mid`
  - `lat_cell_id`, `lon_cell_id`, `p_cell_id`

`DomainSpec` stores the resolved control-volume bounds actually used after snapping and margin application.

### 5.3 Geometry And Weight Outputs

Current geometry outputs:

- `A_horizontal(lat, lon)`
- `A_east(level, lat)`
- `A_west(level, lat)`
- `A_south(level, lon)`
- `A_north(level, lon)`
- `V_cell(level, lat, lon)`

Current weight outputs:

- `W_top(time, lat, lon)`
- optional `W_bottom(time, lat, lon)`
- `W_east(time, level, lat)`
- `W_west(time, level, lat)`
- `W_south(time, level, lon)`
- `W_north(time, level, lon)`
- `W_volume(time, level, lat, lon)`

### 5.4 Budget Output Dataset

`budget.calculate_budget()` currently returns an `xarray.Dataset` containing:

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

This output is the main input to the result-plotting functions.

### 5.5 Run Metadata

Each run writes `run_info.json` with the current structure:

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

## 6. Testing

The current test suite is mixed: parts of it reflect the present implementation closely, and parts of it still carry path assumptions from the earlier `eulerian_heat_budget_surface` repository layout.

Current test files:

- `tests/test_grid.py`
  - covers domain cropping, bounds metadata, horizontal areas, vertical areas, and cell volumes
- `tests/test_weights.py`
  - covers horizontal-face weights, vertical-wall weights, and volume weights
- `tests/test_budget_closure.py`
  - covers mass and heat-advection closure behavior for idealized flows
- `tests/test_run_outputs.py`
  - covers run-path creation and git provenance / metadata serialization
- `tests/test_integrals.py`
  - currently empty

Observed status in this working tree:

- `tests/test_grid.py` and `tests/test_weights.py` pass under:
  `mamba run -n dev_env pytest -q tests/test_grid.py tests/test_weights.py`
- `tests/test_budget_closure.py` also passes when run on its own
- some test files still hard-code the old `eulerian_heat_budget_surface` path
- because the suite mixes old and current import-root assumptions, not every test file cleanly validates this repository when collected together

This means the testing section should be read as a description of current coverage, not as a claim that the whole suite is fully harmonized.

## 7. Current Implementation Notes

- The document describes current behavior, even where the implementation is transitional.
- Commented-out legacy code in `terms.py` and `plot_results.py` is not part of the active API and is not treated as the current design.
- The plotting modules live under `src/`, not under `scripts/`.
- The main runtime entrypoint is `scripts/run_budget.py`, with `src/budget.py` acting as the orchestration layer used by that script.
