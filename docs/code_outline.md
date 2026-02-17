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

#### Layer A — Geometry & Masks

Responsible for:

- Computing grid-cell areas
- Constructing vertical masks using surface pressure
- Applying regional masks

Outputs:

- `cell_area(y,x)`
- `volume_mask(time,p,y,x)`

This layer contains **no physics**, only geometry and domain logic.

------

#### Layer B — Integrals (Pure Mathematical Operators)

Reusable operators:

- `pressure_integral(field)`
- `area_integral(field_2d)`
- `volume_integral_pcoords(field)`

These functions:

- Contain no I/O
- Contain no dataset-specific assumptions
- Accept arrays + coordinate metadata
- Return deterministic outputs

This layer must be independently unit-tested.

------

#### Layer C — Budget Assembly

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
│   └── Eulerian_Heat_Budget_Software_Design.md
│
├── src/eulerian_heat_budget/
│   ├── __init__.py
│   ├── config.py
│   ├── io.py
│   ├── validate.py
│   ├── grid.py
│   ├── masks.py
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

Computes grid metrics:

- `get_horizontal_cell_areas(lat, lon)`
   - 2D horizontal areas on a spherical grid, output in squared meters
- `get_vertical_cell_areas(p, lat/lon)`
   - 2D vertical cell areas. if horizontal dimension in lat, must account for spherical correction.
- `get_cell_volumes(p, lat, lon)`
   - 3D volume in pressure * squared meters. Horizontal dimensions are spherical, while pressure is assumed to be the radial dimension (linear).
- Handles spherical Earth geometry

Must be deterministic and independently testable.

------

### `masks.py`

Constructs:

- `region_mask`
- `volume_mask(time,p,y,x)`
- `area_mask_vertical(time,p,y/x)`
- `area_mask_horizontal(time,y,x)`

Ensures correct truncation at surface pressure.

------

### `integrals.py`

Pure integration operators:

- `pressure_integral(field, mask)`
   - depends on `area_mask_vertical`
- `area_integral(field_2d, cell_area)`
   - depends on `area_mask_horizontal`
- `volume_integral(field, mask, cell_area)`

Implements:

$$
\int_V (\cdot) dV =
\sum_{y,x}
\left[
\int \frac{\text{field}}{g} dp
\right] dA
$$

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

- `time`
- `p`
- `y`
- `x`

------

### Required Coordinates

- `p` [Pa]
- `lat(y)`
- `lon(x)`
- `time`

------

### Required Variables

| Variable | Dimensions   | Units  |
| -------- | ------------ | ------ |
| `T`      | (time,p,y,x) | K      |
| `u`      | (time,p,y,x) | m s⁻¹  |
| `v`      | (time,p,y,x) | m s⁻¹  |
| `omega`  | (time,p,y,x) | Pa s⁻¹ |
| `ps`     | (time,y,x)   | Pa     |

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

1. Center-of-cell vs full-cell interpretation
2. Pressure spacing non-uniformity
3. Mask truncation near topography
4. Vertical differencing of time tendency
5. ERA5 analysis increments (budget non-closure)

All must be explicitly documented in future versions.

------

## 9. Extension Pathways

- Energy-weighted integrals (ρc_vT)
- Moist static energy version
- Lagrangian cross-validation
- Comparison to model energy budgets
- Multi-model ensemble automation

------

## 10. Versioning Plan

Use semantic versioning:

- `0.x`: development
- `1.0`: validated heat budget implementation
- `2.0`: extended moist/static energy version

------

