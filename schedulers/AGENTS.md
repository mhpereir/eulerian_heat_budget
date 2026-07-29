# Alliance Slurm adapter instructions

These instructions apply to the `schedulers/` subtree on
`drac_development_2_staged`. The root `AGENTS.md` remains authoritative for the
scientific contract, shared development workflow, provenance, and global
command-center requirements.

## Establish the execution authority

- Invoke the `alliance-hpc` skill before changing, staging, submitting,
  monitoring, or diagnosing Alliance work.
- Treat `drac_development_2_staged` as the authoritative branch for the staged
  Alliance and Slurm pathway.
- Keep the local Git repository as the development source of truth. Never run
  an agent, edit source, build environments, or perform scientific computation
  on a cluster login node.
- Inspect the intended cluster, account, storage, modules, environment, and
  current scheduler state live. Do not carry Fir connectivity assumptions to
  Rorqual.

## Require a reproducible Slurm environment

- Require explicit `SLURM_ACCOUNT`, `EHB_PYTHON_MODULE`, and `EHB_VENV`
  settings in the submission path. Do not commit personal absolute environment
  paths.
- Use `module purge`, load a complete explicit module stack, then activate the
  declared virtual environment. Print and verify the resolved interpreter.
- On Fir, external access from compute nodes must still be treated as mutable
  and verified. On Rorqual, stage every dependency before submission and never
  download packages or data from a compute job.
- Keep source, inputs, outputs, caches, checkpoints, and logs in explicitly
  selected locations appropriate to their lifecycle. Copy valuable results out
  of `$SLURM_TMPDIR` before exit.

## Maintain Slurm scripts and submission wrappers

- Start scripts with a Bash shebang and `set -euo pipefail`.
- Specify account, walltime, CPU count, memory, and any GPU requirement
  explicitly. Omit a partition unless a verified requirement makes it
  necessary.
- Require a clean named source commit and record it in job provenance.
- Keep environment, input, output, cache, checkpoint, and log paths
  configurable.
- Print job and array IDs, host, commit, interpreter, paths, start time, and end
  time.
- Use bounded arrays for yearly production work and a dependent `afterok`
  consolidation job. Do not submit thousands of individual jobs in a loop.
- Submit through a tracked wrapper that validates the branch, upstream, clean
  state, task bounds, resources, and dependency graph.
- Treat individual legacy `*slurm.sh` scripts as incomplete production
  orchestration when the tracked wrapper, consolidation step, or campaign
  manifest is absent. Do not reconstruct missing production steps manually.

## Stage, monitor, and validate

- Validate source and scripts locally first, then stage only reviewed code and
  small configuration files to an explicit absolute remote project path.
- Never use `rsync --delete` without separate authorization and exact target
  verification.
- Start with a short queued test, inspect logs and measured resources, then
  size production requests from evidence.
- Record every returned Slurm job ID in the global command center immediately,
  including array and dependent job IDs. Monitor initial queue state and logs
  for a few minutes and timestamp the observation.
- Preserve scheduler-neutral `scheduler`, `job_id`, `array_index`, commit,
  environment, resources, paths, and campaign identity in run provenance.
- Do not start the production calculation until every expected yearly shard
  validates and consolidation has published the required campaign catalog and
  summary.
- After terminal scheduler state, inspect `sacct` or equivalent accounting,
  exit status, elapsed time, peak memory, artifacts, coverage, and scientific
  checks before marking the task complete.

## Keep mutations explicit

Read-only cluster, queue, accounting, and log inspection is the default.
Require task authorization before creating remote files or environments,
synchronizing code, submitting or changing jobs, deleting data, or changing
permissions or allocation configuration.
