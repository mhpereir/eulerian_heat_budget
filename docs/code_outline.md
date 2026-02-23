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

2. **Advection (via divergence form)**
   $$
   A(t) = \int_{V(t)} -\nabla \cdot (\mathbf{U}T) dV
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

### Layer A — Geometry & Masks

Responsible for:

- Standardising volume domain for the calculation with `determine_domain()`
- Computing grid-cell areas and volumes (grid.py)
- Constructing 2D (horizontal and vertical) weights for grid-cell-areas and 3D masks for grid-cell-volumes for masking cell elements below the surface

Outputs:

- geometric `cell_area_horizontal(y,x)` and `cell_area_vertical(p,'x' or 'y')`
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

```
eulerian_heat_budget/
│
├── pyproject.toml
│
├── docs/
│   └── code_outline.md
│
├── src/eulerian_heat_budget/
│   ├── __init__.py
│   ├── config.py
│   ├── io.py
│   ├── validate.py
│   ├── grid.py
│   ├── weights.py
│   ├── integrals.py
│   ├── terms.py
│   ├── budget.py
│   └── cli.py
│
├── tests/
│   ├── test_integrals.py
│   ├── test_masking.py
│   └── test_budget_closure.py
│
└── scripts/
    └── run_budget.py
```

------

## 5. File-by-File Description

### `config.py`

Defines:

- Physical constants:
  - `g`
  - `R`
  - `cp`
- Upper boundary pressure
- Lower boundary constraint (either surface or pressure level)
- Region configuration
- Default integration methods

Contains no runtime logic.

------

### `io.py`

Responsibilities:

- Load datasets (ERA5, CMIP6, etc.)
- Harmonize variable names
- Enforce pressure units (Pa)
- Return standardized `xarray.Dataset`

Should not perform calculations.

------

### `validate.py`

Strict schema validation:

- Required variables present
- Required dimensions present
- Pressure monotonic decreasing
- Lat/lon monotonic ascending
- Units consistent
- Time coordinate regular

Raises errors if violated.

This prevents silent scientific errors.

------

### `grid.py`

Computes grid metrics for easy integrals later.

Geometric cell areas:

- `get_horizontal_cell_areas(lat, lon)`
  - 2D horizontal areas on a spherical grid, output in squared meters.
  - horizontal coordinates are treated as full-cell interval starts, so output dims are `(lat_cell, lon_cell)` with shape `(nlat-1, nlon-1)`.
- `get_vertical_cell_areas(p, lat/lon)`
  - 2D vertical side-wall areas for east/west/south/north boundaries.
  - uses full-cell horizontal intervals and level-point pressure coordinates.

Geometric cell volumes:

- `get_cell_volumes(p, lat, lon)`
  - 3D volume in pressure * squared meters.
  - horizontal dimensions are full-cell spherical intervals; pressure thickness is inferred from level centers.
- Handles spherical Earth geometry

Must be deterministic and independently testable.

------

### `weights.py`

Constructs data arrays of weights with the following behaviour:

- 0 if cell is completely below the surface
- 1 if cell is completely above the surface
- fractional value [0-1] if only part of the cell is above the surface

Outputs the following variables:

- `volume_weights(time,p,y,x)`
- `area_weights_vertical(time,p,y/x)`
- `area_weights_horizontal(time,y,x)`

Ensures correct truncation at surface pressure.

------

### `integrals.py`

Pure integration operators:

- `pressure_integral(field, mask)`
  - depends on `area_weights_vertical`
- `area_integral(field_2d, cell_area)`
  - depends on `area_weights_horizontal`
- `volume_integral(field, mask, cell_area)`
  - depends on `volume_weights`

Must not depend on dataset structure.

------

### `terms.py`

Computes individual budget components:

- `compute_storage(T)`
- `compute_advective_term(U, T)`
- `compute_adiabatic_term(omega, T)`
- `compute_diabatic_term(S,A,C)`

May call `integrals.py` internally.

------

### `budget.py`

High-level orchestration:

- Assembles all terms
- Computes time tendency
- Computes residual
- Returns structured output dataset

No I/O inside.

------

### `cli.py`

Command-line interface:

- Parses arguments
- Loads data
- Calls `budget.compute_budget`
- Saves results

------

### `scripts/run_budget.py`

Entry script used on HPC or locally.

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
| `sfp`    | (time,lat,lon)       | Pa     |

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
