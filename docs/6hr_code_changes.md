# 6-Hour Phase Downsampling Variant

Branch: `6_hourly_test_era`

## 1. Purpose

This branch is intended to test how accurately the Eulerian heat-budget calculation performs when the input ERA5 time axis is downsampled from 1-hour intervals to 6-hour intervals.

The incoming data format should remain unchanged. The code should load, standardize, and validate the hourly data as it does now, then create 6-hour phase subsets during preprocessing and run the existing budget calculation on each subset.

The six phase groups are defined by UTC hour modulo 6:

| Phase `r` | UTC hours included |
| --- | --- |
| `0` | 00, 06, 12, 18 |
| `1` | 01, 07, 13, 19 |
| `2` | 02, 08, 14, 20 |
| `3` | 03, 09, 15, 21 |
| `4` | 04, 10, 16, 22 |
| `5` | 05, 11, 17, 23 |

Equivalently, each phase group is:

```text
G_r = {r, r + 6, r + 12, r + 18}, r = 0, 1, 2, 3, 4, 5
```

## 2. Design Goals

- Preserve the existing 1-hour workflow as the default behavior.
- Add a branch-specific 6-hour mode that runs the budget once per selected phase.
- Keep `budget.calculate_budget()` and lower-level physics code operating on normal datasets with a regular `time` coordinate.
- Make the calculation time-step agnostic by relying on actual datetime differences, not hard-coded hourly assumptions.
- Store the combined 6-hour comparison artifact with `phase` and `sample` dimensions instead of a shared `time` dimension.
- Keep per-phase diagnostic plots separate, because the current plotting functions expect a normal `time` coordinate.

## 3. Temporal Sampling

Add a small temporal sampling layer after data loading, standardization, and validation:

```text
phase mask: dataset.time.dt.hour % stride_hours == phase
stride_hours = 6
phase in [0, 5]
```

Recommended helper location: `src/time_utils.py`.

Recommended helper behavior:

- Validate `stride_hours > 0`.
- Validate `0 <= phase < stride_hours`.
- Require the dataset to have a `time` coordinate.
- Select the phase subset using `ds.sel(time=mask)`.
- Verify the selected phase has enough samples for centered time differencing.
- Reuse `require_regular_time()` to verify the resulting phase subset is regularly spaced.

The current budget code should continue to see a valid 1D `time` coordinate inside each phase run. For example, the `r=0` budget run should operate on a dataset whose `time` coordinate is 00, 06, 12, 18, 00, ... at regular 6-hour spacing.

## 4. Runtime Flow

The current single-run workflow in `scripts/run_budget.py` should be refactored into reusable pieces:

1. Parse CLI and build `DomainRequest`, `SurfaceBehaviour`, `DataSourceConfig`, runtime controls, and production options.
2. Load and validate the full hourly source dataset once.
3. Load ARCO benchmark fluxes once when `SourceCfg.kind == "arco_era5"`.
4. Determine whether the run is:
   - existing default single hourly workflow
   - all six 6-hour phases
   - one selected 6-hour phase
5. For each selected phase:
   - subset the loaded source dataset with `time.dt.hour % 6 == phase`
   - subset benchmark fluxes the same way, when present
   - determine `ds_domain`, `ds_halo`, and `DomainSpec` for the phase subset
   - run `budget.calculate_budget()` normally
   - write per-phase plots under a phase-specific directory when plotting is enabled
6. Aggregate phase results only after all requested phase runs complete.
7. Write one combined 6-hour comparison output for all-phase mode, or one phase output for single-phase mode.

The first implementation should avoid changing the internals of `budget.calculate_budget()` unless a hidden 1-hour assumption is found during testing.

## 5. CLI And Runtime Controls

Add branch-specific CLI options:

- `--six-hourly-phases`
  - run all six phase groups: `0, 1, 2, 3, 4, 5`
- `--six-hourly-phase R`
  - run a single phase group
  - useful for debugging and scheduler array jobs

Validation rules:

- `--six-hourly-phases` and `--six-hourly-phase` are mutually exclusive.
- `--six-hourly-phase` must be an integer in `[0, 5]`.
- If neither option is provided, preserve the existing hourly behavior.

The selected temporal sampling mode should be recorded in run metadata and production manifests.

Recommended metadata fields:

```json
{
  "temporal_sampling": {
    "mode": "six_hourly_phases",
    "stride_hours": 6,
    "phase_hours": [0, 1, 2, 3, 4, 5],
    "phase_definition": "UTC hour modulo stride_hours"
  }
}
```

For a single-phase run, `phase_hours` should contain only the selected phase, for example `[3]`.

## 6. Output Design

### 6.1 Per-Phase Budget Runs

Each phase run should initially return the normal budget output with a real `time` coordinate. This keeps the existing derivative, integration, diagnostic, and plotting code compatible.

Per-phase plot directories should be:

```text
plots/<year>/phase_r0/
plots/<year>/phase_r1/
plots/<year>/phase_r2/
plots/<year>/phase_r3/
plots/<year>/phase_r4/
plots/<year>/phase_r5/
```

For ad hoc runs, use the same phase directory naming under the generated run plot directory.

### 6.2 Combined 6-Hour Artifact

The combined output should not use `time` as a dimension coordinate, because the six phase groups do not share the same timestamp vector.

Primary combined dimensions:

- `phase`
- `sample`

Required coordinates/data variables:

- `phase(phase)`
  - integer phase labels: `0, 1, 2, 3, 4, 5`
- `phase_hour(phase)`
  - same values as `phase`; included for clarity
- `sample(sample)`
  - integer sample index within each phase
- `valid_time(phase, sample)`
  - original timestamps from each phase result
- `utc_hour(phase, sample)`
  - UTC hour extracted from `valid_time`
- `valid_sample(phase, sample)`
  - boolean mask for padded slots if phases have unequal sample counts

Time-dependent budget variables should be transformed from:

```text
var(time)
```

to:

```text
var(phase, sample)
```

Phase-scalar values should become:

```text
T_scale(phase)
```

instead of a scalar shared across all phases.

### 6.3 Production Output Paths

Production 6-hour all-phase output should use a distinct filename so it cannot overwrite the standard hourly annual output:

```text
annual/heat_budget_<year>_6hr_phases.nc
```

For single-phase production runs, use:

```text
annual/heat_budget_<year>_6hr_phase_r<R>.nc
```

The existing hourly production output should remain:

```text
annual/heat_budget_<year>.nc
```

## 7. Affected Code Areas

### `src/time_utils.py`

Add temporal phase selection helpers:

- `select_phase_by_utc_hour(ds, *, stride_hours, phase)`
- optionally `validate_phase_selection(time, *, stride_hours, phase)`

These helpers should be dataset-agnostic and usable for both source data and benchmark flux data.

### `src/cli.py`

Add mutually exclusive 6-hour phase options:

- `--six-hourly-phases`
- `--six-hourly-phase`

### `scripts/run_budget.py`

Refactor the current monolithic `main()` into reusable workflow steps:

- build specs and runtime controls
- prepare output paths
- load source and benchmark datasets
- run one budget calculation for a provided source dataset, benchmark dataset, and plot directory
- run one or more 6-hour phase calculations
- aggregate phase outputs

The default path should continue to call the single-run workflow exactly once.

### `src/run_outputs.py`

Add helpers or options for 6-hour output paths:

- all phases: `heat_budget_<year>_6hr_phases.nc`
- single phase: `heat_budget_<year>_6hr_phase_r<R>.nc`

Extend metadata writers so temporal sampling details can be serialized in:

- ad hoc `run_info.json`
- production `production_run.json`

### Plotting Modules

The existing plot functions can remain unchanged for the first implementation. They should receive per-phase datasets that still have a normal `time` coordinate.

Combined phase-output plotting can be deferred unless a comparison plot is required later.

## 8. Edge Cases And Constraints

- Centered time differencing drops the first and last sample of each phase. A phase subset must therefore have at least three timestamps after downsampling.
- Time slicing should happen before phase selection so the selected phases respect `--time-start` and `--time-end`.
- If the input time axis is timezone-naive `datetime64`, treat the values as UTC, consistent with ERA5 conventions.
- If a requested time window begins or ends mid-day, some phases may have one more sample than others. The combined output must support padded samples through `valid_sample`.
- ARCO benchmark fluxes must be downsampled with the same phase selector as the source dataset.
- Existing hourly output and metadata contracts must not change when no 6-hour option is used.

## 9. Test Plan

Unit tests:

- Select each phase from 24 hourly timestamps and verify the expected UTC hours.
- Verify each selected phase is regularly spaced at 6 hours.
- Reject invalid `stride_hours` values.
- Reject invalid phase values.
- Verify selection fails clearly when no `time` coordinate exists.

Workflow tests:

- Mock `scripts/run_budget.py` dependencies and verify `--six-hourly-phases` runs the budget six times.
- Verify `--six-hourly-phase 3` runs the budget once with phase `3`.
- Verify benchmark flux data is phase-selected when present.
- Verify default hourly behavior still runs exactly once when no 6-hour flag is provided.

Aggregation tests:

- Combine synthetic phase results with unequal lengths.
- Verify time-dependent variables become `(phase, sample)`.
- Verify scalar variables become `(phase)`.
- Verify `valid_time`, `utc_hour`, and `valid_sample` are present and correctly shaped.

Time-step agnosticism tests:

- Verify `compute_time_derivative()` gives the same physical rate for equivalent linear 1-hour and 6-hour time series.
- Verify `integrate_rate_over_time()` uses the actual timestep for 6-hour data.

Production-output tests:

- Verify all-phase production output writes to `heat_budget_<year>_6hr_phases.nc`.
- Verify single-phase production output writes to `heat_budget_<year>_6hr_phase_r<R>.nc`.
- Verify standard hourly production output remains `heat_budget_<year>.nc`.
- Verify overwrite protection applies independently to the new 6-hour output files.

## 10. Open Follow-Up Decisions

These choices are not required for the first implementation but may matter after the first successful branch run:

- Whether to add combined phase-aware plots over `(phase, sample)`.
- Whether to compare each phase directly against the full hourly result in the same output file.
- Whether production scheduler scripts should run all phases in one job or use array jobs with one phase per job.
- Whether to expose `stride_hours` as a configurable value beyond this branch's fixed 6-hour experiment.
