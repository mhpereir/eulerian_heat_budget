# Google Cloud Batch adapter instructions

These instructions apply to the `deployment/gcp/` subtree on
`google_development_staged`. The root `AGENTS.md` remains authoritative for the
scientific contract, shared development workflow, provenance, and global
command-center requirements.

## Establish the deployment authority

- Invoke the `ehb-gcp-batch` skill before preparing, changing, submitting,
  resuming, downloading, consolidating, or diagnosing a campaign.
- Read `docs/README.md` completely for the canonical command sequence. Use
  `docs/google-cloud-batch-deployment.md` for architecture, schema, IAM,
  integrity, observability, and recovery details.
- Treat this adapter as staged ARCO acquisition. It publishes yearly cache
  shards and does not execute `scripts/run_budget.py` or produce heat-budget
  NetCDF output.
- Keep a busy checkout untouched. Build production images only from a clean,
  committed source revision.

## Preserve configuration and identity

- Treat `src/config.py::REGIONS` as the canonical named-region registry.
- Normalize the exact campaign, inspect the normalized JSON, and record its
  SHA-256 before submission.
- Bind each campaign ID permanently to one normalized configuration. Use a
  distinct ID for every canary and a new ID whenever any normalized setting
  changes.
- Use only digest-pinned production image URIs. Record the immutable digest
  outside ephemeral build logs.
- Keep stable project, region, bucket, repository, image-name, and service
  account values separate from run-specific campaign IDs, image digests, job
  IDs, job UIDs, cache roots, and output paths.
- Keep credentials and tokens out of repository files. Let `gcloud` own
  authentication.
- Do not edit `requirements-arco.txt` manually. Change
  `requirements-arco.in` and regenerate the locked file with the documented
  dependency workflow.

## Validate locally and through the container gate

- Use `dev_env` for local compilation, unit tests, consolidation checks, and
  small prototypes. Confirm `sys.executable` and required imports.
- Treat `requirements-arco.txt` as the runtime dependency contract for this
  branch. Repair declared dependencies instead of falling back to the base
  environment or bypassing a check.
- Run focused tests and the complete local suite. Run `bash -n` on changed
  shell scripts.
- Preserve the production container gate. Dependency or container changes must
  pass the Zarr smoke test and full test suite before producing an image.

## Follow the production lifecycle

- Verify the active account, project, region, resources, permissions, quotas,
  image digest, and live job state rather than relying on shell history.
- Prefer tracked setup, render, build, submission, and retrieval scripts over
  reconstructed commands.
- Gate every full campaign on a completed, downloaded, consolidated, and opened
  one-year canary.
- Before a billable submission, report campaign identity and hash, image
  digest, task count, parallelism, machine type, disk size, intended job ID,
  bucket prefix, and existing success-marker count.
- Monitor job and task state by job ID. Resolve the immutable job UID for log
  queries and verify the exact expected yearly success markers.
- Resume an unchanged campaign and image under a new job ID. Do not mutate a
  campaign while reusing its ID.
- Download into a fresh POSIX directory with adequate capacity. Resume an
  interrupted transfer with the same storage synchronization command.
- Consolidate directly with the compatible Python environment. Docker and
  Artifact Registry access are not required for local consolidation.

## Track and validate each run

- Immediately record campaign ID and hash, image digest, Batch job ID and UID,
  task count, parallelism, resource profile, bucket prefix, and monitoring
  command in the global command center.
- Monitor initial job and task logs for a few minutes and timestamp the
  observation. Recover recent job IDs before assuming a submission was lost.
- Preserve normalized campaign JSON, rendered job JSON, image digest, shard
  manifests, success markers, and consolidation summary as run provenance.
- Distinguish Batch success from usable scientific data. Require complete
  marker coverage, a validated download, successful consolidation, opened
  catalogs, and aligned calculation settings before marking the task complete.

## Keep mutations explicit

Read-only job, task, log, resource, and Cloud Storage inspection is the
default. Require task authorization before changing APIs, IAM, or cloud
resources; building or pushing images; uploading campaigns; submitting or
retrying jobs; deleting cloud data; starting a large download; or publishing
consolidated metadata.
