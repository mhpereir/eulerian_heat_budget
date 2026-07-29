# Eulerian Heat Budget agent instructions

These instructions apply to every active branch. The repository uses branch
tips to carry deployment-specific adapters, so a file required on one backend
branch may be intentionally absent from another.

## Establish branch authority

Before changing code or preparing a run:

1. Inspect the current branch, upstream, commit, and worktree.
2. Identify the intended execution pathway from the branch and requested task.
3. Read the shared scientific documentation present on that branch.
4. Read backend documentation only when the matching backend component exists.
5. Keep busy or dirty checkouts untouched and use a separate local worktree.

The active branch roles are:

- `production_development`: portable scientific implementation and shared
  architecture, without a required deployment backend.
- `production_development_staged`: Venus and OpenPBS staged retrieval and
  production calculation.
- `drac_development_2_staged`: Digital Research Alliance of Canada and Slurm
  staged retrieval and production calculation.
- `google_development_staged`: Google Cloud Batch staged ARCO acquisition and
  local calculation handoff.

Do not infer that a backend is incomplete merely because another branch's
adapter, scheduler, runbook, lockfile, or deployment directory is absent. Do
not recreate or copy a missing backend component unless the task explicitly
ports that component. Start backend-specific work from its authoritative branch
tip. Start portable scientific work from `production_development` on a
dedicated branch, then port the focused commit to backend tips.

Keep this root `AGENTS.md` synchronized across the active branch tips. Put
narrower instructions beside a backend adapter only when they truly apply to
that subtree.

## Preserve the scientific contract

- Treat `docs/Eulerian Heat Budget - Reformulation.md` as the source of truth
  for physics, definitions, sign conventions, and closure interpretation.
- Read `docs/code_outline.md` before changing the pipeline, data contracts, or
  module responsibilities.
- Do not resolve a disagreement by silently changing the formulation. State
  the discrepancy, trace it through equations and units, and add a regression
  test for the intended behavior.
- Preserve physical units and sign conventions in names, attributes,
  diagnostics, metadata, and tests.
- Physics changes require focused numerical tests, including an analytic or
  limiting case where practical.
- Scheduler and deployment adapters must not redefine scientific terms,
  canonical datasets, staged-cache semantics, or output schemas.

## Respect project boundaries

- Keep scientific implementation in `src/` and staged-cache implementation in
  `src_arco/`.
- Keep command orchestration in `scripts/`.
- Keep schedulers and service adapters outside the scientific modules.
- Keep shared scientific and architectural documentation in `docs/`.
- Keep backend-specific commands and operational details beside their adapter.
- Treat input datasets as read-only.
- Do not commit credentials, environment secrets, raw datasets, caches, logs,
  checkpoints, or generated results.
- Never add personal absolute paths to source, tests, scheduler templates, or
  documentation. Accept paths through CLI arguments, environment variables, or
  ignored project-local configuration.
- Prefer targeted ignore rules. Do not broadly ignore a file type when tests
  need small intentional fixtures of that type.
- Do not manually edit generated changelogs, lockfiles, or deployment
  artifacts. Change their declared inputs and use the documented generator.

## Preserve configuration and provenance

- Use `EHB_DATA_ROOT` or `--local-data-path` for local ERA5 input.
- Use `EHB_OUTPUT_ROOT` or `--output-root` for ad hoc output.
- Use `--production-output-dir` for production campaigns.
- Keep stable deployment values separate from campaign IDs, image digests, job
  IDs, cache roots, and output paths.
- Treat a normalized campaign ID as permanently bound to one configuration.
  Use a new ID when any year, season, domain, vertical boundary, overflow,
  chunking, timeout, or benchmark setting changes.
- Use a distinct campaign ID for every canary.
- Preserve normalized campaign configuration, source commit, environment,
  scheduler or service identity, shard manifests, success markers,
  consolidation summary, and scientific validation as run provenance.
- When the branch provides the staged run-manifest implementation, create
  `production_run.json` atomically at the campaign root before submission.
  Otherwise preserve the backend's documented equivalent. Record the
  authoritative branch and commit, normalized settings, source, paths, runtime
  environment, requested resources, and every retrieval and consolidation job
  pairing.
- Run production work only from a named, clean commit. A queued job must verify
  the expected commit before computation.
- Treat input datasets as read-only. Write small local experiments under
  ignored `tmp/`; write large remote intermediates and results to the storage
  location appropriate to the active backend.
- Preserve or extend the run metadata whenever adding a scheduler, service, or
  execution mode.

## Use the shared local environment

- Use the local Mamba environment named `dev_env` for compilation checks, unit
  and regression tests, local consolidation checks, and small scientific
  prototypes.
- Activate it explicitly with `mamba activate dev_env`, then print or inspect
  `sys.executable` before recording validation results.
- Treat `environment.yml` and any branch-specific locked requirements as the
  dependency contracts. If `dev_env` is missing a required package, repair the
  declared environment rather than silently using the base environment or
  bypassing a check.
- Treat `dev_env` as the first cross-platform compatibility gate, not as proof
  that Venus OpenPBS, Alliance Slurm, or a Google container workflow will
  succeed. Preserve validation on the actual target platform.

Minimal local validation starts with:

```bash
mamba activate dev_env
python -c "import sys; print(sys.executable)"
python -m pytest -q
python scripts/run_budget.py --help
```

## Develop and validate locally

- Make source, documentation, test, and scheduler changes locally first.
- Keep each commit focused enough to review and port safely.
- Run focused tests for the changed scientific, cache, campaign, retrieval,
  manifest, resume, consolidation, scheduler, or deployment behavior.
- Run the complete local suite in the branch's declared environment:

  ```bash
  python -m pytest -q
  ```

- Run `python scripts/run_budget.py --help` before deployment.
- Run `bash -n` on every changed shell script.
- Treat every lint, test, scientific-validation, and flaky-test failure as work
  to resolve, even when it predates the current change.
- Do not claim a real-data workflow is validated from unit tests alone. Record
  the job identity, commit, configuration, environment, paths, and scientific
  checks used.

## Track long-running work

- Invoke the `command-center` skill for qualifying asynchronous operations.
- In addition to the shared schema, record the campaign ID and hash, immutable
  image digest when applicable, backend-specific job and dependency identities,
  retrieval and consolidation relationship, cache and output roots, manifests,
  and success markers.
- Mark the task complete only after expected artifacts, success markers,
  coverage, and scientific checks pass.
- Keep detailed immutable provenance in `production_run.json`, campaign
  manifests, scheduler logs, and scientific validation records beside outputs.

## Route Google Cloud work

Use this pathway only when `deployment/gcp/AGENTS.md` is present, normally on
`google_development_staged`. That nested contract marks the adapter as loaded
on the current branch.

- Invoke the `ehb-gcp-batch` skill.
- Read `deployment/gcp/AGENTS.md` before changing or operating the adapter.
- Read `docs/README.md` completely before preparing, submitting, resuming,
  downloading, or consolidating a campaign.
- Treat `deployment/gcp` as staged ARCO acquisition. It does not execute
  `scripts/run_budget.py` or produce heat-budget NetCDF output.

If the nested contract or the two Google runbooks are absent, this pathway is
not loaded on the current branch. Switch to or branch from
`google_development_staged` rather than treating the absence as a defect.

## Route Alliance and Slurm work

Use this pathway only when `schedulers/AGENTS.md` identifies a Slurm adapter,
normally on `drac_development_2_staged`.

- Invoke the `alliance-hpc` skill.
- Read `schedulers/AGENTS.md` before changing or operating the Slurm adapter.
- Treat the local repository as the development source of truth and the
  cluster as an execution target.
- Submit all computation through Slurm. Do not compute on a login node.

If that nested Slurm contract is absent, switch to or branch from
`drac_development_2_staged` rather than adding cluster assumptions to another
tip.

## Route Venus and OpenPBS work

Use this pathway only when `schedulers/AGENTS.md` identifies an OpenPBS adapter,
normally on `production_development_staged`.

- Invoke the `venus-hpc` skill.
- Read `schedulers/AGENTS.md` and `schedulers/README.md` before changing or
  operating the OpenPBS adapter.
- Treat `production_development_staged` as the only authoritative source branch
  for Venus staged retrieval and production `run_budget` submissions.
- Keep the local Git repository authoritative and use the shared Git remote to
  transport code.
- Submit all tests and scientific computation through OpenPBS, never on the
  login node.

If that nested OpenPBS contract is absent, switch to or branch from
`production_development_staged` rather than adding Venus assumptions to another
tip.

## Definition of done

A change is complete when:

1. The implementation and documentation agree with the scientific contract.
2. The change is based on the authoritative shared or backend branch.
3. Runtime paths and mutable identities are configurable and no machine-local
   runtime value was introduced.
4. Focused tests, the complete suite, shell checks, and CLI smoke tests pass in
   the declared environment.
5. A representative queued or service-backed test is recorded when real backend
   behavior is part of the claim.
6. Generated data remain outside Git and the final diff contains only intended
   project files.
