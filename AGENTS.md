# Agent instructions

These instructions apply to the entire repository. More specific instructions
may be added inside a backend directory when that backend needs extra rules.

## Scientific contract

- Treat `docs/Eulerian Heat Budget - Reformulation.md` as the source of truth
  for the physics, definitions, sign conventions, and closure interpretation.
- Read `docs/code_outline.md` before changing the pipeline or module
  responsibilities.
- Do not resolve a disagreement by silently changing the formulation. State
  the discrepancy, trace it through the equations and units, and add a
  regression test for the intended behavior.
- Preserve physical units and sign conventions in variable names, attributes,
  diagnostics, and tests.
- Physics changes require focused numerical tests, including an analytic or
  limiting case where practical.

## Development workflow

- Make source, documentation, and test changes in the local repository first.
- Start portable work from `production_development` on a dedicated branch.
- Keep a change focused enough to cherry-pick onto active backend tips.
- Run `pytest -q` and `python scripts/run_budget.py --help` before deployment.
- Do not claim a real-data workflow is validated from unit tests alone. Record
  the scheduler job ID, Git commit, configuration, output path, and scientific
  checks used for validation.
- Keep the runtime checkout on a named, clean Git commit. Do not run production
  science from uncommitted source changes.

## Shared local environment

- Use the local Mamba environment named `dev_env` for compilation checks, unit
  and regression tests, and small scientific prototypes.
- Activate it explicitly with `mamba activate dev_env`, then print or inspect
  `sys.executable` before recording validation results.
- Treat `environment.yml` and any backend-specific locked requirements as the
  dependency contracts. Do not resolve a missing package by silently using the
  base environment or by making an unrecorded environment-only change.
- The local `dev_env` is intended to mirror the Python environments used on
  Venus, Alliance HPC, and the Google container image. Use it as the first
  cross-platform compatibility gate, not as proof that a remote scheduler or
  container workflow will succeed.
- Keep platform validation intact: OpenPBS on Venus, Slurm on Alliance, and the
  container build and smoke-test gates for Google.

Minimal local validation starts with:

```bash
mamba activate dev_env
python -c "import sys; print(sys.executable)"
python -m pytest -q
python scripts/run_budget.py --help
```

## Project boundaries

- Keep scientific implementation in `src/` and orchestration in `scripts/`.
- Keep scheduler and service adapters out of the scientific modules.
- Keep general scientific and architectural documentation in `docs/`.
  Backend-specific instructions belong beside the corresponding scheduler or
  deployment adapter.
- Do not duplicate the root `AGENTS.md` with different shared rules on each
  branch. Add a nested `AGENTS.md` only when a backend directory needs narrower
  instructions.
- Do not commit datasets, credentials, environment secrets, caches, logs, or
  generated results.
- Never add a personal absolute path to source, tests, or scheduler templates.
  Accept paths through CLI arguments, environment variables, or an ignored
  project-local configuration file.
- Prefer targeted ignore rules. Do not use a broad file-pattern ignore when a
  small tracked fixture of that type could be scientifically useful.

## Runtime configuration

- Use `EHB_DATA_ROOT` or `--local-data-path` for local ERA5 inputs.
- Use `EHB_OUTPUT_ROOT` or `--output-root` for ad hoc outputs.
- Use `--production-output-dir` for production campaigns.
- Treat input datasets as read-only. Write small local experiments under
  ignored `tmp/`; write large remote intermediates and results to the storage
  location appropriate to the active backend.
- Preserve or extend the run metadata whenever adding a scheduler, service, or
  execution mode.

## Backend routing

For Venus or PBS work:

- Invoke the `venus-hpc` skill.
- Use the Git-first workflow: commit and push locally, then fast-forward a clean
  checkout on Venus.
- Submit compute and data-processing work with `qsub`. Do not run it on the
  Venus login node.
- Avoid broad `rsync` from existing Venus project directories because they may
  contain large untracked datasets.

For Alliance, Fir, Rorqual, Slurm, or DRAC work:

- Invoke the `alliance-hpc` skill.
- Develop locally and sync only source and small configuration files.
- Submit compute and data-processing work with `sbatch`. Do not run it on a
  login node.
- Account for the network difference: Fir compute nodes have internet access;
  Rorqual compute nodes do not.

For Google work:

- Treat Google services as the staged ARCO data-acquisition path unless the
  project scope is explicitly changed.
- Follow the deployment documentation and nested instructions present on the
  Google branch.
- Do not move core heat-budget computation to Google merely because Google
  deployment files are present.

## Definition of done

A change is complete when:

1. The implementation and documentation agree with the physics source of
   truth.
2. Data and output paths are configurable and no machine-local path was added.
3. Relevant unit and regression tests pass in the declared environment.
4. A small real-data test, when required, runs through the target scheduler or
   service and records provenance.
5. Generated data remain outside Git and the final diff contains only intended
   project files.
