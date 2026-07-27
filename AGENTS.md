# Eulerian Heat Budget agent instructions

## Project and branch scope

- Treat `docs/README.md` as the canonical command-oriented runbook for the
  Google Cloud Batch staged ARCO retrieval workflow. Read it completely before
  preparing, submitting, resuming, downloading, or consolidating a campaign.
- Use `docs/google-cloud-batch-deployment.md` for design, schema, IAM, integrity,
  observability, and recovery details.
- Keep the deployment boundary explicit: `deployment/gcp` retrieves and
  publishes yearly staged cache shards. It does not run
  `scripts/run_budget.py` or produce heat-budget NetCDF results.
- Do not apply Venus/OpenPBS or Alliance/Slurm workflows to Google Cloud Batch
  retrieval unless the user separately requests a later HPC calculation.
- Preserve other checkouts. If another checkout is busy or dirty, work from a
  separate clean clone and inspect the busy checkout read-only.

## Source and configuration invariants

- Treat `src/config.py::REGIONS` as the canonical named-region registry used by
  campaign validation and the scientific CLI. Add or change regions there with
  focused tests.
- Treat normalized campaign JSON as immutable under its `campaign_id`. Use a
  new ID when any year, season, domain, vertical-boundary, overflow, chunking,
  timeout, or benchmark-variable setting changes.
- Use a distinct campaign ID for every canary.
- Use only digest-pinned production images of the form
  `...@sha256:<64 lowercase hexadecimal characters>`.
- Keep stable deployment values separate from run-specific campaign IDs, image
  digests, job IDs, job UIDs, cache roots, and output paths.
- Keep credentials and tokens out of repository files.
- Do not manually edit `requirements-arco.txt`; it is the generated,
  hash-locked dependency file. Change `requirements-arco.in` and regenerate it
  with the repository's intended dependency workflow.
- Do not manually edit generated changelogs or generated deployment artifacts.

## Development and validation

- Use the local Mamba environment named `dev_env` for compilation checks, unit
  tests, local consolidation checks, and small prototypes. Activate it
  explicitly and confirm `sys.executable` before recording results.
- The local `dev_env` is intended to mirror the environments used on Venus and
  Alliance HPC and the Python environment built into the Google runtime image.
  It is the first compatibility gate, not a substitute for Cloud Build or Batch
  validation.
- Treat `requirements-arco.txt` as authoritative for this branch. If `dev_env`
  lacks a required package or differs from the lock, repair the declared
  environment rather than silently falling back to base or bypassing a test.
- Keep production image builds on a clean worktree with a committed `HEAD`.
  Never use `--allow-dirty` for production.
- Run focused tests for changed campaign, renderer, retrieval, manifest, resume,
  consolidation, or cache-loader behavior.
- Run the complete local suite from an environment containing the dependencies
  locked in `requirements-arco.txt`:

  ```bash
  mamba activate dev_env
  python -m pytest -q
  ```

- For shell changes, run `bash -n` on every changed shell script.
- For campaign changes, normalize the exact campaign file, inspect the
  normalized JSON, and record the SHA-256 with
  `python -m deployment.gcp.campaign`.
- For container or dependency changes, preserve the Docker build gate. It must
  complete the Zarr smoke test and full pytest suite before producing the
  runtime image.
- Treat every lint, test, scientific-validation, or flaky-test failure as work
  to resolve, even when it predates the current change.

## Cloud operations

- Verify the active `gcloud` account, project, region, resource names, and
  permissions before relying on shell state or prior runs.
- Prefer the tracked scripts in `deployment/gcp` over manually reconstructed
  setup, rendering, build, or submission commands.
- Use read-only job, task, log, resource, and Cloud Storage inspection by
  default.
- Require explicit authorization before enabling APIs, changing IAM or cloud
  resources, building and pushing images, uploading campaigns, submitting or
  retrying jobs, deleting cloud data, or starting large downloads.
- Before submission, report the normalized campaign ID and hash, immutable image
  digest, task count, parallelism, machine type, disk size, job ID, bucket
  prefix, and existing success-marker count.
- Gate every full campaign on a completed, downloaded, consolidated, and opened
  one-year canary.
- Resume with an unchanged campaign and image under a new job ID. Do not mutate
  a campaign while reusing its ID.
- Preserve normalized campaign JSON, rendered job JSON, image digest, shard
  manifests, success markers, and consolidation summary as run provenance.

## Download and calculation handoff

- Download a campaign to a fresh directory on a POSIX filesystem with adequate
  capacity. Resume an interrupted transfer with the same `gcloud storage rsync`
  command rather than mixing in stale campaign content.
- Run `scripts/consolidate_staged_arco_cache.py` directly from the repository
  root with the existing Python environment containing the dependencies locked
  in `requirements-arco.txt`. Local consolidation does not require Docker or
  Artifact Registry access.
- Pass the host campaign directory directly to `--cache-root`. Record the
  repository commit, resolved Python executable, and environment dependency
  state with the campaign provenance.
- Require successful `cache.sqlite` and `consolidation.json` validation before
  exposing the directory as a `staged_arco_cache`.
- Keep the later calculation's domain, vertical bounds, season, years, overflow
  setting, and benchmark-variable setting aligned with the staged campaign.
- Do not request surface variables from this staged-cache format.
