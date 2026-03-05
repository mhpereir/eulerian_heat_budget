# Eulerian Heat Budget — Software Design Document

## 1. Purpose

This project implements a physically consistent, volume-integrated Eulerian heat budget in pressure coordinates, with the intention of calculating the integrated diabatic heating term as the residual:

$$
\frac{d}{dt}\int_{V(t)} T dV =
-\iint_{\partial V_{\text{sides+top}}} T(\mathbf{U}\cdot \hat{n}) dA
+
\int_{V(t)} \left(
\omega\frac{RT}{c_p p}
+
\frac{J}{c_p}
\right) dV
$$

The implementation is designed to:

- Operate on reanalysis or climate model data (e.g., ERA5, CanESM).
- Support regional control volumes (e.g., PNW mask).
- Maintain strict dimensional and unit consistency.
- Be testable using synthetic and reduced datasets.

------

## 2. Mathematical Framework (Pressure Coordinates)

### 2.1 Volume Element

In pressure coordinates:

$$
dV = \frac{1}{g} dp dA
$$

Thus any volume integral becomes:

$$
\int_V (\cdot) dV =
\iint_A
\int_{p_{\text{top}}}^{p_s(x,y,t)}
\frac{(\cdot)}{g} dp dA
$$
where:

- ( $p_s(x,y,t)$ ) = surface pressure
- ( $p_{\text{top}}$ ) = fixed upper boundary
- ( $g$ ) = gravitational acceleration

------

### 2.2 Budget Terms

We compute:

1. **Storage**
   $$
   S(t) = \int_{V(t)} T dV
   \quad \Rightarrow \quad
   \frac{dS}{dt}
   $$

2. **Advection (via surface flux form)**
   $$
   A(t) = \int_{A(t)} T(\mathbf{U}\cdot \hat n) dA
   $$

3. **Adiabatic compression**
   $$
   C(t) = \int_{V(t)} \omega \frac{RT}{c_p p} dV
   $$

4. **Diabatic heating (Residual)**
   $$
   D(t) = S(t) - [A(t)+C(t)]
   $$

------

## 3. Core Design Layers

The implementation is structured in three conceptual layers.

------

### Layer A — Geometry & Weights

Responsible for:

- Standardising volume domain for the calculation with `determine_domain()`
- Computing grid-cell areas and volumes (grid.py)
- Constructing 2D (horizontal and vertical) weights for grid-cell-areas and 3D masks for grid-cell-volumes for masking cell elements below the surface; fractional value of volume above the surface.

Outputs:

- geometric `cell_area_horizontal(y,x)`
- geometric `cell_area_vertical(p,'x' or 'y')`
- geometric `cell_volume(p,y,x)`
- `volume_weights(time,p,y,x)`
- `horizontal_area_weights(time,y,x)`
- `vertical_area_weights(time,p,'x' or 'y')`

This layer contains **no physics**, only geometry and domain logic.

------

### Layer B — Integrals (Pure Mathematical Operators)

Reusable operators:

- `area_integral_vertical(field_2d_vertical)`
- `area_integral_horizontal(field_2d_horizontal)`
- `volume_integral_pcoords(field_3d)`

These functions:

- Contain no I/O
- Contain no dataset-specific assumptions
- Accept arrays + coordinate metadata
- Return deterministic outputs

This layer must be independently unit-tested.

------

### Layer C — Budget Assembly

Responsible for:

- Computing all physical budget terms
- Combining integrals
- Producing diagnostic outputs
- Reporting closure residual

This layer depends on Layers A and B.

------

## 4. Repository Structure

``` #type: ignore
eulerian_heat_budget/
│
├── pyproject.toml
│
├── docs/
│   └── code_outline.md
│
├── src/
│   ├── __init__.py
|   ├── cli.py
│   ├── config.py
│   ├── io.py
│   ├── specs.py
│   ├── validate.py
│   ├── grid.py
│   ├── weights.py
│   ├── integrals.py
│   ├── terms.py
│   ├── budget.py
│   ├── plot_diagnostics.py
│   └── plot_results.py
│
├── tests/
│   ├── test_integrals.py
│   ├── test_grid.py
│   ├── test_weights.py
│   └── test_budget_closure.py
│
└── scripts/
    └── run_budget.py
```

------

## 5. File-by-File Description

### `config.py`

Defines:

- Physical constants (currently: g, R, cp)
- Standard pressure levels (e.g., LEVELS_HPA)
- Default dataset paths / region aliases (project-specific)
- Runtime defaults used by CLI (when flags are not provided)

Contains minimal/no runtime logic.

------

### `specs.py`

Dataclasses that define the project's contracts:

- `DomainRequest`: user intent (bounds, margin, etc.)
- `DomainSpec`: resolved/validated domain (explicit bounds and metadata)

This is the home for “what the domain means” and should stay free of I/O and heavy computation.

------

### `io.py`

Responsibilities:

- Load datasets (ERA5, model data, etc.)
- Harmonize variable names into the canonical internal schema
- Enforce pressure units (Pa) and consistent coordinate names
- Return standardized xarray.Dataset objects
- Should not perform analysis calculations (no integrals, no budgets).

Contract requirement: `io.py` is where any renaming between external conventions (ERA5 variable names) and internal names must happen (e.g., surface pressure → `sp`).

------

### `validate.py`

Strict schema validation:

- Required variables present
- Required dimensions present and in canonical order
- Pressure monotonic decreasing
- Lat/lon monotonic ascending
- Units consistent
- Time coordinate regular

Raises errors if violated.

------

### `grid.py`

Geometry + domain resolution.

Key responsibilities:

- determine_domain(ds, req):
  - crops the dataset to whole grid cells consistent with the requested bounds
  - returns (ds_domain, DomainSpec)
  - constructs and attaches cell-boundary coordinates:
    - lat_start, lat_end, lon_start, lon_end
    - p_start, p_end
  - constructs and attaches cell IDs for bookkeeping (e.g., lat_cell_id, lon_cell_id, p_cell_id)
  - rewrites lat/lon to cell centers after cropping (internal convention)

This module should remain deterministic and independently testable.

------

### `weights.py`

Builds fractional occupancy weights that account for surfaces intersecting the volume.

Current responsibilities (implemented / in-progress):

- Volume occupancy weights on `(time, level, lat, lon)` that represent “fraction of the cell within the control volume”
- Area weights for control-volume faces (horizontal and vertical walls), with fractional masking where the surface intersects the face

These weights are the bridge between “geometry” and later “integrals/budget terms”.

Note: the detailed set of returned arrays and their naming should be documented in the schema section (Section 6) and treated as an API.

------

### `integrals.py`

Pure integration operators (no I/O):

- horizontal/vertical surface integrals
- volume integrals in pressure coordinates
- time differencing helpers for storage terms

Should depend only on numpy/xarray objects and weight arrays, not on CLI/config.

------

### `budget.py`

High-level orchestration (no I/O):

- assembles terms into a single output dataset
- computes residual / closure diagnostics
- plots diagnostics
- exposes a stable programmatic API for scripts

Outputs the four main integral components:

- `dT_dt` (storage)
- `net_advected_heat` (advection)
- `adiabatic_heating` (work/adiabatic)
- `diabatic_heating` (local heat sources, residual)

------

### `terms.py`

Computes individual budget components (calls `integrals.py`):

- storage tendency
- advective flux divergence via control-volume faces
- adiabatic/compressional term(s)
- diabatic as residual (or explicitly if available)

------

### `cli.py`

Command-line interface:

- Parses arguments

------

### `scripts/run_budget.py`

Main script, used on HPC or locally.

------

### `scripts/plot_diagnostics.py`

Plots intermediate diagnostic figures to ensure integrals are working correctly. Called from budget.py

- fig 1: mass continuity test of `net_advected_mass` vs time rate of change of volume `dV_dt`

$$
\frac{d​}{dt}\iiint_{V(t)} ​dV_p +  ​\iint \vec U \cdot n dA_p​​​=0
$$

- fig 2: advection terms
  - top panel: volume of domain;
  - middle panel: advection terms (east, west, south, north, top, and bottom if present)
  - bottom panel: sum of advection terms (`net_heat_advection`)
- fig 3: time series of absolute flux residual

------

### `scripts/plot_results.py`

Plot results of integral calculation. Called from run_budget.py

- fig1:
  - top panel `dT_dt`
  - bottom panel `net_head_advection`, `adiabatic_heating`, `D (diabatic heating, residual term)`

------

## 6. Rigid Data Structure (Required Schema)

### Required Dimensions

- `time`: hourly frequency
- `level`: variable spacing
- `lat`: 2 degree spacing
- `lon`: 2 degree spacing

------

### Required Coordinates

- `time`: same as dimension, 1hr average
- `level`: same as dimension, cell center
- `lat`: same as dimension, cell center
- `lon`: same as dimension, cell center
- `level_start`: cell start
- `level_end`: cell end
- `lat_start`: cell start
- `lat_end`: cell end
- `lon_start`: cell start
- `lon_end`: cell end

Note: in the INPUT data (io.py) horizontal variables follow the full-cell description of data, meaning that the lat/lon grid represents the average over the full grid-cell [x, x+1]. Pressure coordinates represent the variable at each pressure level (horizontally averaged), and `dp` is inferred from level-centered edge extrapolation during integration/geometry calculations. This is standardized in `determine_domain()`.

------

### Required Variables

| Variable | Dimensions           | Units  |
| -------- | -------------------- | ------ |
| `T`      | (time,level,lat,lon) | K      |
| `u`      | (time,level,lat,lon) | m s⁻¹  |
| `v`      | (time,level,lat,lon) | m s⁻¹  |
| `w`      | (time,level,lat,lon) | Pa s⁻¹ |
| `sp`     | (time,lat,lon)       | Pa     |

Future variables: (surface variables)

| Variable | Dimensions        | Units  |
| -------- | ----------------- | ------ |
| `T10m`   | (time,lat,lon)    | K      |
| `u10m`   | (time,lat,lon)    | m s⁻¹  |
| `v10m`   | (time,lat,lon)    | m s⁻¹  |
| `w10m`   | (time,lat,lon)    | Pa s⁻¹ |

### Domain dimensions

| Variable    | Dimensions   | Units  |
| ----------- | ------------ | ------ |
| `lat_min`   | float        | deg    |
| `lat_max`   | float        | deg    |
| `lon_min`   | float        | deg    |
| `lon_max`   | float        | deg    |
| `margin`    | int          | int    |

Note: the domain extent is defined by the input domain extent (from io) minus some margin (measured in array index) to ensure there's a few points outside so extrapolation is never necessary.

------

### Required Physical Assumptions

- Hydrostatic balance implied.
- Pressure coordinate vertical axis.
- Surface is a material boundary.
- No mass flux across lower boundary.

------

## 7. Testing Strategy

### Unit Tests

- Constant field integral
- Analytic pressure integral
- Surface truncation behavior

#### Implemented Grid Unit Tests (`tests/test_grid.py`)

1. `test_determine_latlon_domain_interval_start_semantics`
   - Verifies that `determine_latlon_domain` interprets horizontal coordinates as **full-cell interval starts**, not center points.
   - For `margin = 1`, expects returned bounds to be the first and last valid **cell-start** values after trimming one cell from each side.
   - For an over-large margin, expects a `ValueError`, confirming the function rejects cases where no interior cells remain.

2. `test_horizontal_cell_areas_shape_and_metadata`
   - Verifies output dimensional contract of `get_horizontal_cell_areas`:
     - dims must be exactly `("lat_cell", "lon_cell")`
     - shape must be `(nlat-1, nlon-1)`.
   - Verifies interval metadata coordinates are correct:
     - `lat_start`, `lat_end`, `lon_start`, `lon_end`
     - midpoint helper coordinates `lat_mid`, `lon_mid`.
   - Verifies units are `m2` and all computed values are finite and strictly positive.

3. `test_horizontal_cell_areas_analytic_regular_grid`
   - Validates spherical horizontal area computation against a closed-form analytic reference on a regular grid:
     $$
     A = R^2 |\sin(\phi_n)-\sin(\phi_s)|\,|\lambda_e-\lambda_w|
     $$
   - Expects numerical agreement at tight tolerance (`rtol=1e-12`, `atol=0.0`), confirming correct trigonometric and interval handling.

4. `test_vertical_cell_areas_shape_and_units`
   - Verifies output contract of `get_vertical_cell_areas` for all four side walls:
     - `east(level, lat_cell)`, `west(level, lat_cell)`,
       `south(level, lon_cell)`, `north(level, lon_cell)`.
   - Verifies bbox selection is applied using **interval-start semantics** by checking selected `lat_cell`/`lon_cell` indices and associated start/end metadata.
   - Verifies units are `m*Pa`, values are finite and positive, and `east == west` for identical meridional geometry.

5. `test_vertical_cell_areas_analytic_regular_grid`
   - Validates wall-area magnitudes against analytic formulas on a regular grid:
     - `east/west = dp * (R * dphi)`
     - `south/north = dp * (R * cos(phi_boundary) * dlon)`.
   - Confirms pressure-thickness handling (`dp`) and boundary-latitude dependence are correct.
   - Explicitly checks `south != north` where boundary latitudes differ, ensuring north/south wall geometry is not incorrectly forced symmetric.

6. `test_vertical_cell_areas_irregular_grid_positive`
   - Uses nonuniform pressure and horizontal spacing to verify robustness away from regular grids.
   - Expects all four wall outputs to remain finite and strictly positive under irregular interval widths.

7. `test_vertical_cell_areas_error_paths`
   - Verifies explicit failure behavior for invalid inputs:
     - bbox selects zero cells,
     - non-monotonic latitude coordinate,
     - horizontal coordinate with fewer than two points.
   - Expects `ValueError` for each case, preventing silent geometry misuse.

------

### Integration Tests

- Small spatial subset
- Few pressure levels
- Single time slice

------

### Closure Diagnostics

For real datasets:

- Expect non-zero residual
- Monitor:
  - Relative residual magnitude
  - Sensitivity to resolution
  - Sensitivity to time differencing scheme

------

## 8. Known Numerical Sensitivities

1. Integral will be sensitive to resolution
2. Pressure spacing non-uniformity
3. Mask truncation near topography
4. Vertical differencing of time tendency
5. ERA5 analysis increments (budget non-closure)

All must be explicitly documented in future versions.

------

## 9. Extension Pathways

- Add surface variables (T10m, ux10m, uv10m, uz10m) for improved residual calculation.
- Multi-model ensemble automation

------

## 10. Versioning Plan

Use semantic versioning:

- `0.x`: development
- `1.0`: validated heat budget implementation
- `2.0`: extended moist/static energy version

------
