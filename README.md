# Eulerian Heat Budget

This repository computes a vertically integrated Eulerian atmospheric heat
budget from ERA5 data. It contains the shared scientific implementation plus
optional adapters for three execution environments:

- Alliance HPC clusters, especially Fir and Rorqual, using Slurm
- UVic Climate Lab servers through Venus, using PBS
- Google services for staged ARCO ERA5 acquisition

The Google workflow is currently a data-staging backend, not a second
implementation of the heat-budget calculation.

## Scientific authority

[Eulerian Heat Budget - Reformulation](<docs/Eulerian Heat Budget - Reformulation.md>)
is the source of truth for the physics, definitions, signs, and closure
interpretation that the code must reproduce. If code, tests, or other
documentation disagree with it, treat that disagreement as a defect to
investigate rather than silently changing the formulation.

[code_outline.md](docs/code_outline.md) describes the software pipeline,
module responsibilities, internal data contracts, outputs, and current
implementation status.

## Repository layout

- `src/` - scientific calculations, data contracts, I/O, validation, and
  plotting
- `scripts/run_budget.py` - main command-line entry point
- `tests/` - unit and regression tests
- `schedulers/` - scheduler adapters present on the current branch
- `deployment/` - optional service-specific deployment code on branches that
  need it
- `docs/` - scientific, architectural, and backend-specific documentation

Large input datasets and generated results do not belong in Git.

## Environment

The shared development environment targets Python 3.12. Create it with Mamba:

```bash
mamba env create -f environment.yml
mamba activate ehb-dev
```

The dependency ranges cover the validated Venus `dev_env` package set and the
Python 3.12 wheels currently available on Fir. On Alliance systems, create the
virtual environment from the cluster-provided wheels as directed by the
`alliance-hpc` skill instead of solving a Conda environment on a login node.

Run the local checks before deploying:

```bash
pytest -q
python scripts/run_budget.py --help
```

## Data and output paths

Machine-specific absolute paths are configuration, not source code.

- `EHB_DATA_ROOT` points to the directory containing local ERA5 NetCDF inputs.
  `--local-data-path` overrides it for one run.
- `EHB_OUTPUT_ROOT` sets the root for ad hoc results. It defaults to
  `results/` in this checkout, and `--output-root` overrides it for one run.
- Production campaigns use the explicit `--production-output-dir` argument.

The local ERA5 directory is expected to contain `T.nc`, `ux.nc`, `uy.nc`,
`uz.nc`, and `sfp.nc`. Surface-variable runs additionally require
`surface_temperature.nc`, `surface_ux.nc`, and `surface_uy.nc`.

For example:

```bash
export EHB_DATA_ROOT=/path/to/ERA5_zg_PNW
export EHB_OUTPUT_ROOT=/path/to/ehb-results

python scripts/run_budget.py \
  --data-source local_era5 \
  --region pnw_hotz \
  --time-start 1941-06-01T00:00:00 \
  --time-end 1941-06-07T00:00:00
```

Dataset-backed calculations must run through a scheduler on HPC systems, not
on a login node.

## Execution backends

Backend adapters are allowed to differ, but the scientific modules and their
tests remain shared.

| Backend | Deployment model | Scheduler or service |
| --- | --- | --- |
| Venus | Develop locally, commit and push, then fast-forward the clean checkout on Venus | PBS via `qsub` |
| Fir or Rorqual | Develop locally, sync code only, then submit from the remote project checkout | Slurm via `sbatch` |
| Google | Stage ARCO ERA5 data using the deployment documentation present on Google branches | Google services |

Use the `venus-hpc` or `alliance-hpc` Codex skill for the corresponding remote
workflow. Fir compute nodes have internet access; Rorqual compute nodes do not.
Branch-specific scheduler or deployment details should live beside their
adapter, not in the shared scientific modules.

## Branch model

`production_development` is the integration base for portable scientific and
project-contract changes. Backend tips such as
`production_development_staged`, `drac_development_2_staged`, and
`google_development_staged` may contain different scheduler or deployment
adapters.

Make shared changes on a dedicated branch from the integration base, validate
them there, and then cherry-pick or merge the focused commits into each active
backend tip. Do not merge an entire divergent backend branch merely to acquire
a shared contract change.

Contributor and agent operating rules are in [AGENTS.md](AGENTS.md).
