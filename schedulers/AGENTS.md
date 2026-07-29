# Venus OpenPBS adapter instructions

These instructions apply to the `schedulers/` subtree on
`production_development_staged`. The root `AGENTS.md` remains authoritative for
the scientific contract, shared development workflow, provenance, and global
command-center requirements.

## Establish the production authority

- Invoke the `venus-hpc` skill and read `schedulers/README.md` before changing,
  deploying, submitting, resuming, or diagnosing this workflow.
- Treat `production_development_staged` as the only authoritative source branch
  for Venus staged retrieval and production `run_budget` submissions.
- Before submission, require a clean, named production checkout whose `HEAD`,
  configured upstream, and live `origin/production_development_staged` tip are
  the same commit. Integrate and push feature work before using it for
  production.
- Never fetch, pull, switch, edit, stash, reset, clean, or resolve conflicts in
  a checkout referenced by a queued or running job. Create a new
  commit-specific checkout for a new source revision.
- Keep editable checkouts below `~/eulerian-heat-budget/development/`,
  commit-specific production checkouts below
  `~/eulerian-heat-budget/production/`, and production data below
  `~/eulerian-heat-budget/campaign-data/`.

## Preserve the scheduler boundary

- Develop and validate locally. Never run an agent or edit source on Venus.
- Submit tests, retrieval, consolidation, and calculation through OpenPBS.
  Never run scientific computation on the Venus login node.
- Use Git for source. Use narrowly scoped `rsync` only for explicitly selected
  input or result paths, never for a mixed repository root.
- Keep staged caches, `run_budget` outputs, and logs outside `PROJECT_ROOT`
  under the region-nested
  `campaign-data/{staged-zarr,run-budget,logs}/<region>/` layout.
- Use `${VENUS_MAMBA_ENV:-dev_env}` unless a tracked project contract explicitly
  selects another environment. Print and verify the resolved interpreter in
  every job.

## Maintain PBS scripts and submission wrappers

- Start scripts with a Bash shebang and `set -euo pipefail`.
- Request `select`, `ncpus`, memory, and walltime explicitly. Do not use
  `#PBS -V`; initialize the environment explicitly.
- Require `PROJECT_ROOT` and `EXPECTED_COMMIT`, then verify that the checkout is
  clean and at the expected commit before computation.
- Keep input, output, cache, checkpoint, and log paths configurable.
- Print the PBS job ID, host, commit, interpreter, paths, start time, and end
  time.
- Use bounded arrays for yearly work. On Venus, use a standard
  `afterok:<array-parent-id>` dependency for consolidation, not
  `afterokarray`.
- Submit through the tracked wrappers. Do not reconstruct production `qsub`
  commands manually when a wrapper owns preflight, paths, dependencies, or
  provenance.

## Preserve campaign identity and validate outputs

- Create or validate `production_run.json` atomically before staged production
  submission. Record the normalized campaign identity, source commit,
  environment, resources, paths, and every retrieval and consolidation job
  pairing.
- Use a new campaign ID whenever normalized scientific or storage settings
  change. Resume an unchanged campaign without overwriting previous job
  history.
- Record every returned PBS ID in the global command center immediately,
  including array notation and dependent consolidation IDs. Monitor initial
  task state and logs for a few minutes and timestamp the observation.
- Do not start production calculation until every expected yearly shard and
  success marker validates and the dependent consolidation job has published
  the campaign catalog successfully.
- After terminal scheduler state, validate exit status, coverage, manifests,
  checksums, output artifacts, and scientific expectations before marking the
  task complete.

## Keep mutations explicit

Read-only repository, queue, resource, and log inspection is the default.
Require task authorization before pushing, deploying, transferring data,
creating remote files or environments, submitting or altering jobs, deleting
data, or changing permissions.
