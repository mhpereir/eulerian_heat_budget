---
description: Review code changes for physics consistency with the Eulerian heat budget formulation
allowed-tools: Bash, Read, Grep, Glob
---

Review the current code changes (staged, unstaged, or a specific file given in `$ARGUMENTS`) for physics consistency with the Eulerian heat budget framework. This project computes three budget components — advection, adiabatic heating, and diabatic heating (residual) — from 3D wind and temperature fields on a pressure-coordinate grid.

If `$ARGUMENTS` names a file or function, review only that scope. Otherwise, review all uncommitted changes (`git diff HEAD`).

## Theoretical Reference

The full derivation lives in `docs/Eulerian Heat Budget - Reformulation.md`. If you are uncertain about any sign convention, boundary condition, or the anomaly decomposition, read that document before making a judgment call. The summary below is a condensed working reference — defer to the source document when in doubt.

## Theoretical Framework (summary)

The budget equation in integral form is:

```
d<T>/dt = -(1/V) ∮ T'(U · n̂) dA + (1/V) ∫ ω R T / (cp P) dV + (1/V) ∫ J/cp dV
```

where:
- `<T>` is the volume-averaged temperature: `<T> = (1/V) ∫ T dV`
- `T' = T - <T>` is the spatial anomaly (the advection term operates on T', NOT full T)
- `U = (u, v, ω)` with ω (omega) in Pa/s, positive downward
- The diabatic term D is computed as a residual: `D = S + A - C` where S=storage, A=advection, C=adiabatic
- Storage S = `d/dt ∫ T dV`, which decomposes as `<T> dV/dt + V d<T>/dt`

The volume V(t) is time-varying when the bottom boundary follows surface pressure. The Reynolds Transport Theorem surface terms cancel with the surface advection term because there is no mass transfer across the surface boundary (ω = v_b at surface).

Mass closure: `δM = ∮ (U · n̂) dA + dV/dt = 0` (in theory). The heat advection uncertainty is estimated as `δH ≈ T_scale × δM`, where `T_scale = RMSE(T')`.

## Checklist — apply each item to the code under review

### 1. Units
- [ ] Pressure is in Pa throughout (not hPa, not mbar). The standardization in io.py converts on load; new code must not reintroduce hPa.
- [ ] Temperature is in K (not °C).
- [ ] Physical constants (`g`, `cp`, `R_value`, `R_earth`) are imported from `src/config.py`, never hardcoded locally.
- [ ] Cell areas are in m² (horizontal) or m·Pa (vertical walls). Volumes are in m²·Pa. Do not mix metric and pressure-coordinate volumes.
- [ ] Time derivatives use seconds as the time unit (timedelta64 → seconds conversion).

### 2. Sign conventions
- [ ] ω (omega / `w` variable) is positive downward (increasing pressure). This means upward motion has negative ω.
- [ ] Wall normals follow the convention in `terms.py` wall_sign dict:
  - west: -1 (normal points west, into decreasing lon)
  - east: +1 (normal points east, into increasing lon)
  - south: -1 (normal points south, into decreasing lat)
  - north: +1 (normal points north, into increasing lat)
  - top: -1 (normal points upward, into decreasing pressure)
  - bottom: +1 (normal points downward, into increasing pressure)
- [ ] A positive net_heat_advection means net heat is being advected INTO the domain (warming).
- [ ] The adiabatic term `ω R T / (cp P)` must preserve the sign of ω — do not negate it.
- [ ] The residual: `D = S + A - C`. Verify the signs match: storage = -advection + adiabatic + diabatic.

### 3. Dimension ordering and coordinate conventions
- [ ] All 4D fields: `(time, level, lat, lon)`.
- [ ] All 3D surface fields: `(time, lat, lon)`.
- [ ] `level` coordinate is in Pa, monotonically DESCENDING (TOA first = smallest pressure first... but level values decrease with index since pressure increases downward — actually: level is sorted so that isel(level=0) is the HIGHEST pressure = bottom. Check: in this codebase, level is descending meaning isel(level=-1) is the largest pressure value? No — verify against validate.py which enforces level must be monotonically descending, meaning level[0] > level[1], i.e., highest pressure (near surface) first.
- [ ] `lat` is monotonically ascending. `lon` is monotonically ascending.
- [ ] When selecting boundary faces: east/west walls vary over `(level, lat)`, north/south walls over `(level, lon)`, top/bottom faces over `(lat, lon)`.

### 4. Anomaly formulation (critical for advection)
- [ ] The advection integral must use `T' = T - <T>`, not full T. This is what makes the `<T> dV/dt` term cancel with part of the advection integral, reducing sensitivity to mass closure error.
- [ ] When `test_constant_T=True`, T is NOT replaced with `<T>` — instead T is left as-is (no anomaly subtraction), so that the advection captures the full mass-closure error for diagnostic purposes.
- [ ] Surface temperature variables (T2m) must also be converted to anomalies when `use_surface_variables=True`: `T2m' = T2m - <T>`.
- [ ] The storage term on the LHS must correspondingly use `V d<T>/dt` (not `d/dt ∫T dV`) when advection uses anomalies. In the code this is `dT_dt_2 = d_dt_T - <T> * dV_dt`.

### 5. Boundary handling
- [ ] No mass flux across the surface boundary — the RTT cancellation requires `ω|_surface = v_b`. Code must not add a "surface" face to the advection integral.
- [ ] Top face is at `DomainSpecs.zg_top_pressure` (a fixed pressure level).
- [ ] Bottom face depends on `DomainSpecs.zg_bottom`:
  - `"surface_pressure"`: no explicit bottom face; weights handle the time-varying boundary via occupancy fractions.
  - `"pressure_level"`: explicit bottom face at `DomainSpecs.zg_bottom_pressure`.
- [ ] `allow_bottom_overflow`: when True, volume/area weights can exceed 1.0 for the lowest layer if surface pressure exceeds the grid bottom. This accounts for atmosphere below the lowest model level.
- [ ] Halo cells (1-cell pad around domain) are required for face-centered velocity interpolation. Advection face values are averages of adjacent halo cells, NOT extrapolations.

### 6. Integration correctness
- [ ] Area integrals sum over exactly 2 non-time dimensions. Volume integrals sum over exactly 3 non-time dimensions. (Enforced in `integrals.py` but new code could bypass it.)
- [ ] Top/bottom faces use `A_horizontal` (m²). East/west/north/south faces use `A_{wall}` (m·Pa).
- [ ] Weights are applied multiplicatively: `integrand = field × cell_area × weight`.
- [ ] Face-centered quantities use midpoint averaging: `u_face = 0.5 * (u[i] + u[i+1])` from halo grid. NOT one-sided values.

### 7. Mass closure and error estimation
- [ ] The mass closure diagnostic is: `δM = dV/dt + net_mass_advection`.
- [ ] Heat advection error estimate: `δH = (dV/dt + net_mass_advection) × T_scale`.
- [ ] `T_scale` in the normal (non-test) case is `sqrt(mean((T - <T>)²))` — the RMSE of the spatial anomaly.
- [ ] `T_scale` in the constant-T test case is `mean(<T>)` — the mean domain-average temperature.
- [ ] New terms must not break the residual balance: if a new physical term is added, it must appear in both the budget assembly and the diabatic residual calculation.

### 8. Numerical caution
- [ ] Centered finite differences for time derivatives lose the first and last time step — downstream code must handle the shortened time axis.
- [ ] Large intermediate arrays should remain as Dask lazy arrays until `.compute()` is called. Avoid premature `.values` or `.compute()` on 4D fields.
- [ ] Float64 casting (`astype("float64")`) should be applied to integrated (reduced) quantities, not to full 4D fields (memory).

## Output format

For each issue found, report:
1. **File and line** where the issue occurs
2. **Which rule** from the checklist is violated
3. **What's wrong** — be specific about the physics implication
4. **Suggested fix** — concrete code change

If no issues are found, say so explicitly. Do not invent problems.
