# Full-column benchmarking strategy

## 1. Purpose and scope

This document defines the scientific benchmarking strategy for the
full-column Eulerian heat budget. It extends the existing ERA5 benchmark of
temperature-anomaly advection, \(\mathcal{H}'\), to the adiabatic conversion and
diabatic residual terms.

The first implementation phase is deliberately restricted to diagnostics that
ERA5 publishes as **instantaneous, full-column, analysis-compatible fields**.
The three new benchmark variables are:

- `vithe`: vertical integral of thermal energy
- `viec`: vertical integral of energy conversion
- `vithed`: vertical integral of divergence of thermal-energy flux

Model-level forecast-mean tendencies are out of scope for this phase. They are
valuable for attributing the physical parameterizations that contribute to
diabatic heating, but they describe time means over a forecast trajectory.
They are not interchangeable with instantaneous analyzed state diagnostics.
Their use should be investigated in a later, explicitly forecast-based
benchmark.

This strategy follows the formulation in
[Eulerian Heat Budget - Reformulation.md](./Eulerian%20Heat%20Budget%20-%20Reformulation.md).
It does not require reverting from advection of \(T'\) to advection of the full
temperature \(T\). Instead, all three comparisons are placed in the
volume-average temperature framework.

## 2. Conventions and definitions

Let the full atmospheric pressure-coordinate volume be

\[
V(t) = \int_A \int_0^{p_s} dp\,dA,
\]

and define

\[
\langle T\rangle(t)
  = \frac{1}{V(t)}\int_A\int_0^{p_s}T\,dp\,dA,
\qquad
T' = T-\langle T\rangle.
\]

The project works in pressure-volume temperature units. Every integrated
heating term therefore has units
\(\mathrm{K\,m^2\,Pa\,s^{-1}}\). Positive transport and heating increase the
temperature content of the domain.

Define:

\[
\begin{aligned}
\mathcal{M}_{\mathrm{in}}
  &= -\oint_{\partial V}\mathbf{U}\boldsymbol{\cdot}\hat{\mathbf n}\,dA,
\\
\delta\mathcal{M}
  &= \mathcal{M}_{\mathrm{in}}-\frac{dV}{dt},
\\
\mathcal{H}'
  &= -\oint_{\partial V}
       T'\mathbf{U}\boldsymbol{\cdot}\hat{\mathbf n}\,dA,
\\
\mathcal{C}
  &= \int_A\int_0^{p_s}
       \omega\frac{RT}{c_p p}\,dp\,dA,
\\
\mathcal{D}
  &= \int_A\int_0^{p_s}\frac{J}{c_p}\,dp\,dA.
\end{aligned}
\]

Here \(\mathcal{H}'\) is positive into the domain. With ERA5's pressure
vertical velocity convention, \(\omega>0\) for descent, so
\(\mathcal{C}>0\) represents compressional warming and
\(\mathcal{C}<0\) represents expansion cooling.

The volume-average budget, retaining the measured mass-closure residual, is

\[
\boxed{
V\frac{d\langle T\rangle}{dt}
  = \mathcal{H}' + \mathcal{C} + \mathcal{D}
    + \langle T\rangle\delta\mathcal{M}.
}
\]

This equation makes the mass-closure dependence explicit instead of assuming
\(\delta\mathcal{M}=0\). It also distinguishes two useful definitions of the
diabatic residual.

The physical full-temperature residual is

\[
\boxed{
\mathcal{D}_{\mathrm{phys}}
  = V\frac{d\langle T\rangle}{dt}
    -\mathcal{H}'-\mathcal{C}
    -\langle T\rangle\delta\mathcal{M}.
}
\]

The current workflow intentionally defines a volume-average residual without
assigning the mass-closure term to diabatic heating:

\[
\boxed{
\mathcal{D}_0
  = V\frac{d\langle T\rangle}{dt}-\mathcal{H}'-\mathcal{C},
}
\]

so

\[
\mathcal{D}_0
  = \mathcal{D}_{\mathrm{phys}}
    +\langle T\rangle\delta\mathcal{M}.
\]

In the current output schema,

\[
\mathtt{diabatic\_term}=\mathcal{D}_{0,\mathrm{code}},
\qquad
\mathtt{residual\_heat}
  = \langle T\rangle\delta\mathcal{M}_{\mathrm{code}},
\]

and therefore

\[
\mathcal{D}_{\mathrm{phys,code}}
  = \mathtt{diabatic\_term}-\mathtt{residual\_heat}.
\]

Neither definition should silently replace the other. The primary benchmark
must compare the workflow-defined \(\mathcal{D}_0\) on both sides. The
mass-corrected \(\mathcal{D}_{\mathrm{phys}}\) comparison is a secondary test
of the full-temperature equation and may be more sensitive to inconsistencies
in reconstructed mass transport.

## 3. ERA5 full-column variables

ERA5 lists these fields in Table 6, "surface and single level parameters:
vertical integrals and total column: instantaneous." All three are available
for analysis and forecast records. This phase must retrieve the analysis
record and preserve metadata that identifies the product, stream, type, grid,
time, and parameter ID.

| Role | CDS variable | Short name | Param ID | ERA5 units | Definition used here |
|---|---|---:|---:|---:|---|
| Thermal storage | `vertical_integral_of_thermal_energy` | `vithe` | 162060 | \(\mathrm{J\,m^{-2}}\) | \(E_T=(c_p/g)\int_0^{p_s}T\,dp\) |
| Adiabatic conversion | `vertical_integral_of_energy_conversion` | `viec` | 162064 | \(\mathrm{W\,m^{-2}}\) | \(C_E=(1/g)\int_0^{p_s}(RT\omega/p)\,dp\) |
| Heat-flux divergence | `vertical_integral_of_divergence_of_thermal_energy_flux` | `vithed` | 162083 | \(\mathrm{W\,m^{-2}}\) | \(\nabla_H\boldsymbol{\cdot}[(c_p/g)\int_0^{p_s}T\mathbf v\,dp]\) |

The explicit integral definitions are documented in the ERA-Interim archive
and use the same ECMWF local parameter family retained by ERA5. The live ERA5
table is the authority for present availability, names, units, record type,
and parameter IDs.

`viec` is the reversible pressure-work conversion term relevant to this heat
budget. In the temperature equation it is compressional warming or expansion
cooling. The same pressure work appears with the complementary sign in the
mechanical-energy budget. It is not a measure of irreversible dissipation of
kinetic energy into heat.

## 4. Converting ERA5 diagnostics to project units

For a horizontal ERA5 field \(X\), let

\[
\mathcal{I}_A[X] = \int_A X\,dA
\]

use the same physical target domain and cell-area treatment as the main
calculation. ERA5 energy fields are converted to the project's
pressure-volume temperature units by \(g/c_p\).

### 4.1 Storage

\[
\mathcal{S}_{\mathrm{ERA5}}(t)
  = \frac{g}{c_p}
    \frac{d}{dt}\mathcal{I}_A[\mathtt{vithe}](t).
\]

The derivative must use the same centered stencil and the same output times as
the code storage term. The two storage estimates should also be compared
directly:

\[
\frac{g}{c_p}\mathcal{I}_A[\mathtt{vithe}]
\quad\text{versus}\quad
V\langle T\rangle.
\]

This is a prerequisite check. A mismatch here indicates inconsistent
constants, area weights, time alignment, vertical coverage, surface-pressure
treatment, or regridding.

### 4.2 Adiabatic conversion

\[
\boxed{
\mathcal{C}_{\mathrm{ERA5}}
  = \frac{g}{c_p}\mathcal{I}_A[\mathtt{viec}].
}
\]

This benchmark directly tests the model-level integration of
\(\omega RT/(c_p p)\). Its sign must be tested using the explicit integral,
not inferred only from the descriptive variable name.

### 4.3 Diabatic residual

The full-column thermal-energy equation represented by the three ERA5 fields
is

\[
\frac{\partial E_T}{\partial t}
  + \mathtt{vithed}
  = \mathtt{viec} + Q,
\]

so the ERA5 diabatic residual per unit area is

\[
Q_{\mathrm{ERA5,res}}
  = \frac{\partial\mathtt{vithe}}{\partial t}
    +\mathtt{vithed}-\mathtt{viec}.
\]

The domain-integrated benchmark in project units is

\[
\boxed{
\mathcal{D}_{\mathrm{ERA5,res}}
  = \frac{g}{c_p}\mathcal{I}_A
    \left[
      \frac{\partial\mathtt{vithe}}{\partial t}
      +\mathtt{vithed}-\mathtt{viec}
    \right].
}
\]

This is the ERA5 physical full-temperature residual. To express it in the
workflow's volume-average framework, first define the ERA5 mass-closure
residual using ERA5 full-column mass transport and a consistently differenced
full-column mass:

\[
\delta\mathcal{M}_{\mathrm{ERA5}}
  = \mathcal{M}_{\mathrm{ERA5}}
    -\left(\frac{dV}{dt}\right)_{\mathrm{ERA5}}.
\]

The ERA5 counterpart of the workflow-defined residual is then

\[
\boxed{
\mathcal{D}_{0,\mathrm{ERA5}}
  = \mathcal{D}_{\mathrm{ERA5,res}}
    +\langle T\rangle\delta\mathcal{M}_{\mathrm{ERA5}}.
}
\]

The definition-matched primary residual comparison is therefore

\[
\boxed{
\mathtt{diabatic\_term}
\quad\text{versus}\quad
\mathcal{D}_{0,\mathrm{ERA5}}.
}
\]

The secondary physical full-temperature comparison is

\[
\boxed{
\mathtt{diabatic\_term}-\mathtt{residual\_heat}
\quad\text{versus}\quad
\mathcal{D}_{\mathrm{ERA5,res}}.
}
\]

This ERA5 residual is internally useful because storage, flux divergence, and
conversion are all full-column diagnostics from the same ERA5 system. That
self-consistency removes many vertical reconstruction errors from the
benchmark side. It does not make the comparison exact: the project still
reconstructs terms from pressure-level fields, interpolation and grid
operations can differ, analyzed winds need not close the discrete mass budget,
and an instantaneous flux at the center time is not identical to a
time-mean flux over the storage interval.

The secondary comparison is not assumed to correlate better. If the
pressure-level reconstruction and ERA5 full-column diagnostic disagree about
mass transport, subtracting
\(\langle T\rangle\delta\mathcal{M}_{\mathrm{code}}\) can add noise even while
making the code expression formally consistent with the full-temperature
equation.

## 5. Existing advection benchmark in the same framework

ERA5 provides full-temperature horizontal heat fluxes and mass fluxes:

\[
\mathbf F_T
  = \frac{c_p}{g}\int_0^{p_s}T\mathbf v\,dp,
\qquad
\mathbf F_M
  = \frac{1}{g}\int_0^{p_s}\mathbf v\,dp.
\]

After line integration around the horizontal boundary and conversion to
project units, let \(\mathcal{H}_{\mathrm{ERA5}}\) and
\(\mathcal{M}_{\mathrm{ERA5}}\) be positive into the domain. The anomaly
transport benchmark is

\[
\boxed{
\mathcal{H}'_{\mathrm{ERA5}}
  = \mathcal{H}_{\mathrm{ERA5}}
    -\langle T\rangle\mathcal{M}_{\mathrm{ERA5}}.
}
\]

This is already implemented as
`benchmark_heat_flux_net_lateral_prime`. The identity

\[
\mathcal{H}
  = \mathcal{H}' + \langle T\rangle\mathcal{M}_{\mathrm{in}}
\]

shows why retaining \(T'\) is appropriate. The benchmark should test
\(\mathcal{H}'\) directly and should separately expose mass-flux closure,
rather than mixing the much larger reference-temperature transport into the
primary advection comparison.

### 5.1 Interpretation of the existing PNW benchmark

The current PNW Bartusek full-atmosphere diagnostics provide an important
constraint on how the diabatic benchmark should be interpreted:

| Diagnostic | Figure | Pearson correlation |
|---|---|---:|
| Calculated versus ERA5 net mass flux | `fig1.1_benchmark_vs_calculated_mass_flux.png` | 0.704 |
| Two calculated forms of \(dV/dt\) | `fig1.2_benchmark_vs_calculated_dV_dt.png` | 1.000 |
| Calculated versus ERA5 \(\mathcal{H}'\) | `fig5.2_benchmark_vs_calculated_heat_flux_lateral_prime.png` | 0.997 |
| Calculated versus ERA5 full-temperature \(\mathcal{H}\) | `fig5.3_benchmark_vs_calculated_heat_flux_total.png` | 0.876 |

For code-minus-ERA5 differences,

\[
\Delta\mathcal{H}'
  = \Delta\mathcal{H}
    -\langle T\rangle\Delta\mathcal{M}.
\]

The very small \(\Delta\mathcal{H}'\), together with the more scattered
full-temperature heat and mass transports, suggests

\[
\Delta\mathcal{H}
  \simeq \langle T\rangle\Delta\mathcal{M}.
\]

Thus the \(\mathcal{H}'\) agreement strongly validates transport of temperature
anomalies, but it does not establish that
\(\delta\mathcal{M}_{\mathrm{code}}\) and
\(\delta\mathcal{M}_{\mathrm{ERA5}}\) agree. The anomaly transformation can
remove the large reference-temperature transport associated with a
mass-flux disagreement.

The \(dV/dt\) panel also requires careful interpretation. In the current code,
`dV_dt` is derived from discretized cell volumes and `dV_dt_true` is derived
from surface pressure. Their near-perfect agreement validates these two code
paths. It is not yet a comparison with an independently archived ERA5 mass
tendency.

Centered finite differencing can contribute to a nonzero
\(\delta\mathcal{M}\), because an instantaneous mass flux is compared with an
approximation to the instantaneous volume tendency. It cannot explain the
scatter in the direct mass-flux comparison, which contains no \(dV/dt\).
Other candidates include pressure-level reconstruction, vertical truncation
at the moving surface, horizontal interpolation, boundary sampling, and
differences from ERA5's native model-level integrations.

The mass-consistent ERA5-derived product in Reference 4 diagnoses the
full-column mass imbalance and adjusts the analyzed wind at every timestep
before constructing its budget terms. That procedure is additional evidence
that unadjusted analyzed fields should not be assumed to satisfy the discrete
mass budget exactly.

Consequently, the new benchmark must calculate and compare

\[
\delta\mathcal{M}_{\mathrm{code}}
\quad\text{and}\quad
\delta\mathcal{M}_{\mathrm{ERA5}},
\]

as well as their heat-scaled forms. Agreement of \(\mathcal{H}'\) alone cannot
be used as evidence that the heat-scaled mass residuals are interchangeable.

## 6. Benchmark hierarchy

The comparisons should be evaluated in the following order. A higher-level
comparison must not be interpreted before its prerequisites pass.

| Level | Comparison | What it isolates | Required supporting checks |
|---:|---|---|---|
| 0 | Metadata, coordinates, units, constants and time semantics | Data-contract errors | Param IDs, analysis record, full-column coverage, grid and timestamps |
| 1 | \((g/c_p)\int_A\mathtt{vithe}\,dA\) vs. \(V\langle T\rangle\) | Storage normalization | Area weights, \(g\), \(c_p\), \(p_s\), vertical coverage |
| 2 | ERA5 mass flux, \(dV/dt\), \(\delta\mathcal{M}\), and \(\langle T\rangle\delta\mathcal{M}\) vs. their calculated counterparts | Closure and boundary geometry | Positive-into-domain signs, all lateral faces, moving lower boundary |
| 3 | \(\mathcal{H}'_{\mathrm{ERA5}}\) vs. \(\mathcal{H}'_{\mathrm{code}}\) | Temperature-anomaly advection | Levels 0 to 2 |
| 4 | \((g/c_p)\int_A\mathtt{viec}\,dA\) vs. \(\mathcal{C}_{\mathrm{code}}\) | Adiabatic pressure work | Level 1, sign test, \(\omega\) units and convention |
| 5 | ERA5 thermal equation closure from `vithe`, `vithed`, `viec` | Internal consistency of benchmark fields | Centered derivative and temporal-resolution study |
| 6 | \(\mathcal{D}_{0,\mathrm{ERA5}}\) vs. `diabatic_term` | Workflow-defined diabatic residual | Levels 0 to 5 and ERA5-side mass-closure transformation |
| 7 | \(\mathcal{D}_{\mathrm{ERA5,res}}\) vs. `diabatic_term - residual_heat` | Physical full-temperature diabatic residual | Levels 0 to 6 and explicit code-side mass correction |

Two additional residual series should be retained as sensitivity diagnostics:

\[
\mathtt{diabatic\_term}
\quad\text{versus}\quad
\mathcal{D}_{\mathrm{ERA5,res}},
\]

and

\[
\mathtt{residual\_heat}
  = \langle T\rangle\delta\mathcal{M}.
\]

The first intentionally compares different definitions and shows the size of
the ERA5 mass-closure contribution. The second must be reported separately
for code and ERA5 rather than being hidden inside a single score. Whether the
physical comparison at Level 7 improves or degrades agreement is an empirical
result, not an acceptance requirement.

## 7. Time alignment and sampling

`vithe`, `viec`, and `vithed` are classified as instantaneous ERA5
parameters. "Forecast mean" in other ERA5 tables is a temporal mean over a
forecast interval, not a mean over ensemble members. Ensemble means use
separate ensemble-related products and metadata. Forecast-mean model-level
tendencies are deferred here because their forecast interval and trajectory
must be matched explicitly before comparison with analyzed states.

For the instantaneous full-column benchmark:

1. Retrieve hourly analysis fields at identical valid times.
2. Use the project's centered finite difference for `vithe`.
3. Compare only on the centered output time axis.
4. Evaluate one-hour and coarsened two-hour or three-hour calculations from
   the same pilot window.
5. Treat convergence with timestep as a required diagnostic. Do not silently
   tune a tolerance to absorb time-sampling error.

## 8. Hard-data validation contract

The benchmark implementation is not complete until the repository contains a
small, intentional real-ERA5 regression fixture and tests that use it. Large
raw downloads, production caches and credentials must not be committed.

The proposed committed layout is:

```text
tests/
  data/
    full_column_benchmark/
      README.md
      manifest.json
      era5_full_column_benchmark.nc
  test_full_column_benchmark.py
scripts/
  build_full_column_benchmark_fixture.py
```

The exact paths may change during implementation review, but the separation of
source-building code, provenance, compact data and tests is required.

### 8.1 Pilot cases

The pilot should include at least two short hourly windows:

- a dynamically active region and time with substantial ascent, descent and
  lateral heat transport
- a comparatively quiescent region and time with weaker tendencies

Each window must include enough endpoint times for centered derivatives.
Where practical, add a large-area or global case to test cancellation and
mass closure at a different spatial scale. Case selection must be based on
computed signal amplitudes, not on visual preference alone.

### 8.2 Required provenance

The fixture manifest and README must record:

- ERA5 dataset and product type, including `reanalysis`
- retrieval date, valid times, domain bounds, grid and file format
- CDS variable names, short names, parameter IDs and units
- the exact retrieval request or a canonical serialized equivalent
- \(g\), \(c_p\), derivative stencil and boundary sign convention
- source-file checksums and committed fixture checksum
- fixture-building script version and repository commit
- every reduction, crop, cast, regrid or compression operation
- Copernicus licence, acknowledgement and citation requirements

The fixture should contain the smallest data needed to reproduce the tested
calculations. Expected scalar time series may be stored alongside the reduced
fields, but tests must recompute them from the fields where doing so exercises
the production calculation.

### 8.3 Test layers

| Layer | Data | Purpose | Tolerance policy |
|---|---|---|---|
| Algebraic unit tests | Synthetic arrays | Signs, factors of \(g/c_p\), area integration, centered derivatives, mass closure and residual identities | Exact or floating-point roundoff only |
| Fixture regression | Committed real ERA5 subset | Detect any change in reproduced benchmark series | Tight deterministic tolerances based on dtype and library reproducibility |
| Scientific comparison | Fixture plus model-level reconstruction | Measure agreement of \(\mathcal{H}'\), \(\mathcal{C}\), both \(\delta\mathcal{M}\) estimates, and both definitions of \(\mathcal{D}\) | Empirical absolute and normalized thresholds from the pilot |
| Retrieval smoke test | Fresh ERA5 request, opt-in or scheduled | Detect upstream schema, metadata or access changes | Metadata invariants plus statistical scientific bounds |

Required algebraic assertions include:

\[
\begin{aligned}
\mathcal{H}'_{\mathrm{ERA5}}
  &= \mathcal{H}_{\mathrm{ERA5}}
     -\langle T\rangle\mathcal{M}_{\mathrm{ERA5}},\\
\mathcal{C}_{\mathrm{ERA5}}
  &= (g/c_p)\mathcal{I}_A[\mathtt{viec}],\\
\mathcal{D}_{\mathrm{ERA5,res}}
  &= (g/c_p)\mathcal{I}_A[
     \partial_t\mathtt{vithe}+\mathtt{vithed}-\mathtt{viec}],\\
\delta\mathcal{M}_{\mathrm{ERA5}}
  &= \mathcal{M}_{\mathrm{ERA5}}
     -(dV/dt)_{\mathrm{ERA5}},\\
\mathcal{D}_{0,\mathrm{ERA5}}
  &= \mathcal{D}_{\mathrm{ERA5,res}}
     +\langle T\rangle\delta\mathcal{M}_{\mathrm{ERA5}},\\
\mathcal{D}_{0,\mathrm{code}}
  &= \mathtt{diabatic\_term},\\
\mathcal{D}_{\mathrm{phys,code}}
  &= \mathtt{diabatic\_term}-\mathtt{residual\_heat}.
\end{aligned}
\]

### 8.4 Deriving stringent tolerances

Scientific tolerances must not be guessed before the pilot. They must be
derived from the observed distribution across all selected cases and times,
then frozen in the tests with the evidence recorded in the fixture README.

For each term, report at least:

- absolute bias and mean absolute error
- root-mean-square error
- error normalized by benchmark RMS and by benchmark interpercentile range
- correlation
- least-squares slope and intercept
- maximum absolute error
- sign agreement for samples whose magnitude exceeds a documented
  noise floor
- the change in every metric when moving from the definition-matched
  \(\mathcal{D}_0\) comparison to the physical full-temperature comparison

The initial tolerance study must separate:

1. deterministic numerical error from data encoding and floating-point
   operations
2. temporal discretization error
3. horizontal regridding and boundary-sampling error
4. vertical integration and surface-pressure error
5. analyzed-state mass-closure error

The committed regression tolerances should be close to numerical
reproducibility. Scientific tolerances should be the tightest thresholds that
all justified pilot cases pass, with a small documented stability margin.
They must combine an absolute floor with a scale-aware criterion so that
near-zero periods neither dominate relative error nor receive an unlimited
absolute allowance.

A tolerance may be relaxed only with a documented scientific explanation,
updated pilot statistics and explicit review. A change in expected fixture
values is a benchmark update, not routine test maintenance.

## 9. Implementation sequence

1. Add shared ERA5 mappings and retrieval support for `vithe`, `viec` and
   `vithed`, preserving analysis and instantaneous metadata.
2. Extend staged-cache schemas and coverage validation so missing benchmark
   variables fail before calculation.
3. Add isolated conversion functions for storage, adiabatic conversion and
   the ERA5 physical and workflow-defined diabatic residuals.
4. Add output variables with formulas, units, signs, source parameter IDs and
   time semantics in attributes, including separate code and ERA5
   \(\delta\mathcal{M}\) and heat-scaled residuals.
5. Run the real-data pilot and publish the complete metric table.
6. Freeze the compact fixture, provenance and empirically justified
   tolerances.
7. Add plots and production diagnostics only after the numerical tests pass.

Forecast-mean model-level diabatic tendencies remain future work. That phase
must document the forecast initialization, step interval, accumulation or
mean convention, valid-time alignment and any discontinuity introduced by
joining successive forecasts.

## 10. References

1. ECMWF, [ERA5 data documentation, especially Table 6: instantaneous
   vertical integrals and total-column
   parameters](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5%2Bdata%2Bdocumentation).
2. Berrisford et al. (2011), [The ERA-Interim archive, Version
   2.0](https://www.ecmwf.int/sites/default/files/elibrary/2011/8174-era-interim-archive-version-20.pdf),
   Section 4.2 and the explicit vertically integrated parameter definitions.
3. ECMWF, [IFS Documentation CY41R2, Part III: Dynamics and numerical
   procedures](https://www.ecmwf.int/sites/default/files/elibrary/2016/79696-ifs-documentation-cy41r2-part-iii-dynamics-and-numerical-procedures_1.pdf).
4. Copernicus Climate Change Service, [Mass-consistent atmospheric energy and
   moisture budget data derived from ERA5: Product User
   Guide](https://confluence.ecmwf.int/spaces/CKB/pages/260571271/Mass-consistent%2Batmospheric%2Benergy%2Band%2Bmoisture%2Bbudget%2Bdata%2Bfrom%2B1979%2Bto%2Bpresent%2Bderived%2Bfrom%2BERA5%2Breanalysis%2BProduct%2BUser%2BGuide),
   for an independent example of mass adjustment and budget-validation
   practice.
