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
- Run production work only from a named, clean commit. A queued job must verify
  the expected commit before computation.

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

## Route Google Cloud work

Use this pathway only when `deployment/gcp` is present, normally on
`google_development_staged`.

- Invoke the `ehb-gcp-batch` skill.
- Read `docs/README.md` completely before preparing, submitting, resuming,
  downloading, or consolidating a campaign.
- Use `docs/google-cloud-batch-deployment.md` for architecture, schema, IAM,
  integrity, observability, and recovery details.
- Treat `deployment/gcp` as staged ARCO acquisition. It does not execute
  `scripts/run_budget.py` or produce heat-budget NetCDF output.
- Prefer tracked deployment scripts over reconstructed commands.
- Validate and normalize the exact campaign and record its SHA-256.
- Use only digest-pinned production images.
- Gate a full campaign on a completed, downloaded, consolidated, and opened
  one-year canary.
- Resume an unchanged campaign and image under a new job ID.
- Verify the active account, project, region, resource names, permissions,
  quotas, image digest, and current job state live.
- Require explicit authorization before changing APIs, IAM, or cloud resources;
  building or pushing images; uploading campaigns; submitting or retrying jobs;
  deleting cloud data; or starting a large download.

If `deployment/gcp` or the two Google runbooks are absent, this pathway is not
loaded on the current branch. Switch to or branch from
`google_development_staged` rather than treating the absence as a defect.

## Route Alliance and Slurm work

Use this pathway only for Alliance execution, normally on
`drac_development_2_staged`.

- Invoke the `alliance-hpc` skill.
- Treat the local repository as the development source of truth and the
  cluster as an execution target.
- Make and validate all changes locally. Never edit source or run an agent on a
  cluster.
- Submit all computation through Slurm. Do not compute on a login node.
- Use tracked Slurm scripts and explicit account, walltime, CPU, memory, module,
  environment, input, output, checkpoint, and log settings.
- Omit a partition unless a verified requirement makes one necessary.
- Use arrays with bounded concurrency for yearly production work.
- On Fir, compute nodes currently have internet access. On Rorqual, compute
  nodes do not, so stage every dependency before submission.
- Start with a short queued test before production, inspect resource use, and
  adjust future requests from evidence.
- Keep large transient data in suitable cluster storage and retrieve only the
  outputs needed locally.
- Require explicit authorization before synchronizing code, creating remote
  files or environments, submitting or changing jobs, deleting data, or
  changing permissions.

When the Slurm shard workflow is present, read `schedulers/README.md`. Do not
start the production calculation until every expected yearly shard has
validated and consolidation has published `campaign.json`, `cache.sqlite`, and
`consolidation.json` at the campaign root.

If the Slurm adapters are absent, switch to or branch from
`drac_development_2_staged` rather than adding cluster assumptions to another
tip.

## Route Venus and OpenPBS work

Use this pathway only for Venus execution, normally on
`production_development_staged`.

- Invoke the `venus-hpc` skill.
- Keep the local Git repository authoritative and use the shared Git remote to
  transport code.
- Inspect legacy Venus work read-only before adopting it. Stop if Venus has
  uncommitted or unpushed work.
- Commit and push the exact local branch before fast-forward deployment to a
  clean Venus checkout.
- Never edit source, run an agent, resolve Git conflicts, stash, reset, or clean
  on Venus.
- Submit all tests and scientific computation through OpenPBS, never on the
  login node.
- Use tracked PBS scripts with explicit resources, environment initialization,
  input, output, provenance, and log paths.
- Use `${VENUS_MAMBA_ENV:-dev_env}` unless the project explicitly tracks a
  different environment contract.
- Use Git for source. Use narrowly scoped `rsync` only for explicitly selected
  data or results, never for a mixed repository root.
- Start with a queued smoke test and inspect its logs and resource use before
  production.
- Require explicit authorization before pushing, deploying, transferring,
  creating remote files or environments, submitting or altering jobs, deleting
  data, or changing permissions.

When the PBS shard workflow is present, read `schedulers/README.md`. Require a
successful dependent consolidation job before the production calculation
consumes the campaign root.

If the PBS adapters are absent, switch to or branch from
`production_development_staged` rather than adding Venus assumptions to another
tip.

## Definition of done

A change is complete when:

1. The implementation and documentation agree with the scientific contract.
2. The change is based on the authoritative shared or backend branch.
3. Paths and mutable identities are configurable and no machine-local value was
   introduced.
4. Focused tests, the complete suite, shell checks, and CLI smoke tests pass in
   the declared environment.
5. A representative queued or service-backed test is recorded when real backend
   behavior is part of the claim.
6. Generated data remain outside Git and the final diff contains only intended
   project files.
