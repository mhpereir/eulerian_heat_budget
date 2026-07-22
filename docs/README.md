# Google Cloud Batch production runbook

This is the command-oriented runbook for staging the Eulerian heat-budget input
data with Google Cloud Batch, downloading the completed campaign, and building
the local consolidated cache catalog. For the design and implementation details,
see [Google Cloud Batch deployment](google-cloud-batch-deployment.md).

> [!IMPORTANT]
> The current Google Cloud Batch deployment stages yearly ARCO ERA5 cache
> shards. It does not execute `scripts/run_budget.py` and does not produce the
> heat-budget NetCDF results. A successful Batch campaign must be downloaded and
> consolidated before the existing offline production calculation can use it.

The commands below assume Bash and are intended to be run from the repository
root in one shell session.

## 1. Prerequisites

The operator needs:

- a billing-enabled Google Cloud project;
- permission to enable APIs, create the resources in this runbook, and change
  their IAM policies;
- the Google Cloud CLI with an authenticated account;
- Git and Python 3 on the build and submission host;
- Docker on the consolidation host if using the recommended image-based
  consolidation command;
- a POSIX filesystem with enough free space for the downloaded campaign; and
- a clean Git worktree with a committed `HEAD` for a production image build.

The setup script grants its submitter Batch Job Editor, Service Account User on
the runtime service account, and Storage Object User on the campaign bucket. It
does not grant Logging Viewer or local Artifact Registry pull access. An
operator who needs task logs must already have `roles/logging.viewer`. A
principal pulling the private image for local consolidation must already have
Artifact Registry Reader or an equivalent permission.

Authenticate and define the deployment values:

```bash
cd ~/work/google_cloud/eulerian_heat_budget
set -euo pipefail

gcloud auth login

export PROJECT_ID="eulerian-heat-budget"
export REGION="us-central1"
export BUCKET="eulerian-heat-budget-arco-staged-us-central1"
export REPOSITORY="ehb-batch"
export IMAGE_NAME="arco-retrieval"
export SERVICE_ACCOUNT_NAME="ehb-batch-retrieval"

gcloud config set project "${PROJECT_ID}"
gcloud auth list --filter=status:ACTIVE
```

Skip `gcloud auth login` when the intended account is already active, such as
in an authenticated Cloud Shell. If an administrator is preparing resources
for another submitter, also define a full IAM principal:

```bash
export SUBMITTER="user:operator@example.com"
```

## 2. Create or reconcile the Google Cloud resources

Run this once for a new deployment and rerun it after IAM or resource drift:

```bash
./deployment/gcp/setup_batch_resources.sh \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --bucket "${BUCKET}" \
  --repository "${REPOSITORY}" \
  --service-account "${SERVICE_ACCOUNT_NAME}"
```

To grant submission access to `SUBMITTER` instead of the active `gcloud`
account, append:

```bash
./deployment/gcp/setup_batch_resources.sh \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --bucket "${BUCKET}" \
  --repository "${REPOSITORY}" \
  --service-account "${SERVICE_ACCOUNT_NAME}" \
  --submitter "${SUBMITTER}"
```

The script is intentionally safe to rerun. It fails instead of silently using
a bucket in a different region or a non-Docker Artifact Registry repository.

If the operator does not inherit log-view and image-pull permissions from a
broader role, an administrator can grant the two narrowly scoped roles used by
the monitoring and local consolidation commands:

```bash
export OPERATOR_PRINCIPAL="user:operator@example.com"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member "${OPERATOR_PRINCIPAL}" \
  --role roles/logging.viewer \
  --condition=None

gcloud artifacts repositories add-iam-policy-binding "${REPOSITORY}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --member "${OPERATOR_PRINCIPAL}" \
  --role roles/artifactregistry.reader
```

## 3. Create and validate campaign files

Keep production campaign files outside a dirty working tree, or commit them
before building. Create a full production campaign from the tracked example:

```bash
export CAMPAIGN_CONFIG_DIR="${HOME}/ehb-campaigns"
mkdir -p "${CAMPAIGN_CONFIG_DIR}"

export CAMPAIGN_FILE="${CAMPAIGN_CONFIG_DIR}/pnw-1940-2025.json"
cp deployment/gcp/campaign.example.json "${CAMPAIGN_FILE}"
"${EDITOR:-vi}" "${CAMPAIGN_FILE}"
```

At minimum, replace `campaign_id` and review every year, season, domain, and
staging value. There are no scientific fallbacks to the PBS production settings.
Validate the file, inspect its normalized form, and capture its ID:

```bash
python3 -m deployment.gcp.campaign \
  "${CAMPAIGN_FILE}" \
  --normalize-to /tmp/ehb-production-campaign.json

python3 -m json.tool /tmp/ehb-production-campaign.json

export CAMPAIGN_ID="$(python3 -m deployment.gcp.campaign \
  "${CAMPAIGN_FILE}" \
  --print-campaign-id)"

python3 -m deployment.gcp.campaign \
  "${CAMPAIGN_FILE}" \
  --print-sha256
```

Create a separate one-year canary. It must have a different `campaign_id` from
the full campaign because a campaign ID is bound to one normalized
configuration:

```bash
export CANARY_FILE="${CAMPAIGN_CONFIG_DIR}/pnw-canary.json"
cp "${CAMPAIGN_FILE}" "${CANARY_FILE}"
"${EDITOR:-vi}" "${CANARY_FILE}"

python3 -m deployment.gcp.campaign \
  "${CANARY_FILE}" \
  --normalize-to /tmp/ehb-canary-campaign.json

export CANARY_ID="$(python3 -m deployment.gcp.campaign \
  "${CANARY_FILE}" \
  --print-campaign-id)"
```

In the canary file, set `years.start` and `years.end` to the same representative
year and use a unique canary ID.

## 4. Build and capture an immutable image

The production build refuses a dirty worktree. Commit all intended source
changes and confirm that the following command prints nothing:

```bash
git status --short
test -z "$(git status --porcelain)"
```

Build with Cloud Build and retain the output:

```bash
export BUILD_LOG="${TMPDIR:-/tmp}/ehb-cloud-build-$(date -u +%Y%m%d-%H%M%S).log"

./deployment/gcp/build_arco_image.sh \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --repository "${REPOSITORY}" \
  --image "${IMAGE_NAME}" \
  | tee "${BUILD_LOG}"

export IMAGE_URI="$(sed -n 's/^IMAGE_URI=//p' "${BUILD_LOG}" | tail -n 1)"
test -n "${IMAGE_URI}"
printf '%s\n' "${IMAGE_URI}"
```

`IMAGE_URI` must end in `@sha256:` followed by a 64-character hexadecimal
digest. The submission code rejects mutable tags such as `latest`.

## 5. Submit and validate the one-year canary

Use an explicit job ID so the monitoring commands can reuse it:

```bash
export CANARY_JOB_ID="${CANARY_ID:0:47}-$(date -u +%Y%m%d-%H%M%S)"

./deployment/gcp/submit_arco_batch.sh \
  --campaign "${CANARY_FILE}" \
  --image-uri "${IMAGE_URI}" \
  --job-id "${CANARY_JOB_ID}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --bucket "${BUCKET}" \
  --service-account "${SERVICE_ACCOUNT_NAME}" \
  --parallelism 1 \
  --boot-disk-gb 40
```

Monitor the job and its task:

```bash
gcloud batch jobs describe "${CANARY_JOB_ID}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}"

gcloud batch tasks list \
  --job "${CANARY_JOB_ID}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}"
```

Read the task logs by resolving the immutable Batch job UID:

```bash
export CANARY_JOB_UID="$(gcloud batch jobs describe "${CANARY_JOB_ID}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --format='value(uid)')"

gcloud logging read \
  "logName=\"projects/${PROJECT_ID}/logs/batch_task_logs\" AND labels.job_uid=\"${CANARY_JOB_UID}\"" \
  --project "${PROJECT_ID}" \
  --limit 200 \
  --order asc
```

After the job reaches `SUCCEEDED`, verify that the success marker exists:

```bash
gcloud storage ls \
  "gs://${BUCKET}/campaigns/${CANARY_ID}/shards/year=*/_SUCCESS.json" \
  --project "${PROJECT_ID}"
```

Download and consolidate the canary before releasing the full campaign. Use a
fresh directory so stale local shard directories cannot contaminate validation:

```bash
export CANARY_CACHE_ROOT="${PWD}/data/${CANARY_ID}"
test ! -e "${CANARY_CACHE_ROOT}"
mkdir -p "${CANARY_CACHE_ROOT}"

gcloud storage rsync \
  "gs://${BUCKET}/campaigns/${CANARY_ID}" \
  "${CANARY_CACHE_ROOT}" \
  --project "${PROJECT_ID}" \
  --recursive

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --entrypoint python \
  --mount type=bind,src="${CANARY_CACHE_ROOT}",dst=/data \
  "${IMAGE_URI}" \
  scripts/consolidate_staged_arco_cache.py --cache-root /data

test -f "${CANARY_CACHE_ROOT}/cache.sqlite"
test -f "${CANARY_CACHE_ROOT}/consolidation.json"
```

Using the host UID and GID is important because the consolidator atomically
writes both root metadata files into the bind-mounted directory.

## 6. Submit the full production campaign

Do this only after the canary has completed, downloaded, and consolidated
successfully. Consolidation opens each unique cataloged Zarr tile as part of
validation:

```bash
export JOB_ID="${CAMPAIGN_ID:0:47}-$(date -u +%Y%m%d-%H%M%S)"

./deployment/gcp/submit_arco_batch.sh \
  --campaign "${CAMPAIGN_FILE}" \
  --image-uri "${IMAGE_URI}" \
  --job-id "${JOB_ID}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --bucket "${BUCKET}" \
  --service-account "${SERVICE_ACCOUNT_NAME}" \
  --parallelism 5 \
  --boot-disk-gb 40
```

The default resource profile runs at most five `e2-standard-2` workers at once,
with one year per task and one task per VM. Increase parallelism only after
checking regional CPU, instance, and persistent-disk quotas. Increase the boot
disk only if measured scratch usage requires it. The renderer enforces a 30 GB
minimum.

Use the same monitoring commands with `JOB_ID`:

```bash
gcloud batch jobs describe "${JOB_ID}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}"

gcloud batch tasks list \
  --job "${JOB_ID}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}"

export JOB_UID="$(gcloud batch jobs describe "${JOB_ID}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --format='value(uid)')"

gcloud logging read \
  "logName=\"projects/${PROJECT_ID}/logs/batch_task_logs\" AND labels.job_uid=\"${JOB_UID}\"" \
  --project "${PROJECT_ID}" \
  --limit 500 \
  --order asc
```

List published success markers at any time:

```bash
gcloud storage ls \
  "gs://${BUCKET}/campaigns/${CAMPAIGN_ID}/shards/year=*/_SUCCESS.json" \
  --project "${PROJECT_ID}"
```

## 7. Resume a failed production campaign

First inspect the failed task and logs. After correcting an external problem,
submit the same campaign and image under a new job ID:

```bash
export RETRY_JOB_ID="${CAMPAIGN_ID:0:47}-$(date -u +%Y%m%d-%H%M%S)"

./deployment/gcp/submit_arco_batch.sh \
  --campaign "${CAMPAIGN_FILE}" \
  --image-uri "${IMAGE_URI}" \
  --job-id "${RETRY_JOB_ID}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --bucket "${BUCKET}" \
  --service-account "${SERVICE_ACCOUNT_NAME}" \
  --parallelism 5 \
  --boot-disk-gb 40
```

Every task checks its yearly prefix. A valid shard with `_SUCCESS.json` is
skipped. An incomplete prefix is removed and rebuilt. Do not change the campaign
file while reusing its campaign ID. Use a new campaign ID for any parameter
change.

## 8. Download and consolidate production data

Wait for the Batch job to reach `SUCCEEDED`, then download to a fresh local
directory on a POSIX filesystem:

```bash
export CACHE_ROOT="${HOME}/work/google_cloud/ehb-campaigns/data/${CAMPAIGN_ID}"
test ! -e "${CACHE_ROOT}"
mkdir -p "${CACHE_ROOT}"

gcloud storage rsync \
  "gs://${BUCKET}/campaigns/${CAMPAIGN_ID}" \
  "${CACHE_ROOT}" \
  --project "${PROJECT_ID}" \
  --recursive
```

The transfer runs through the host executing `gcloud storage rsync`. For a large
campaign, run it from a host with reliable bandwidth, adequate local disk, and
enough time to complete or resume the transfer.

If the transfer is interrupted, keep the partial directory and rerun only the
`gcloud storage rsync` command. The fresh-directory check is for the first
download, not for a transfer resume.

Authenticate Docker to Artifact Registry and run the exact production image
with its entrypoint overridden:

```bash
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --entrypoint python \
  --mount type=bind,src="${CACHE_ROOT}",dst=/data \
  "${IMAGE_URI}" \
  scripts/consolidate_staged_arco_cache.py --cache-root /data

test -f "${CACHE_ROOT}/cache.sqlite"
test -f "${CACHE_ROOT}/consolidation.json"
python3 -m json.tool "${CACHE_ROOT}/consolidation.json"
```

Consolidation is successful only after every expected year, success marker,
manifest, checksum, SQLite catalog, Zarr time coordinate, and hourly timestamp
has passed validation. The command atomically writes:

- `${CACHE_ROOT}/cache.sqlite`, the combined catalog with paths into the nested
  yearly shard directories; and
- `${CACHE_ROOT}/consolidation.json`, the campaign and per-shard validation
  summary.

The command is resumable and may be rerun after a partial local transfer. It
does not rewrite the Zarr stores.

Optionally publish the two small combined metadata files so future downloads
already contain them:

```bash
gcloud storage cp \
  "${CACHE_ROOT}/cache.sqlite" \
  "${CACHE_ROOT}/consolidation.json" \
  "gs://${BUCKET}/campaigns/${CAMPAIGN_ID}/" \
  --project "${PROJECT_ID}"
```

## 9. Hand the cache to the production calculation

The consolidated directory is now the `staged_arco_cache` root expected by the
scientific calculation:

```bash
export STAGED_CACHE_ROOT="${CACHE_ROOT}"
```

Use this value in `schedulers/production_run_cli_settings.sh` or override it in
the environment used to submit `schedulers/schedule_run_budget_production.sh`.
For the intended production calculation, keep the campaign domain, vertical
bounds, season, years, overflow setting, and benchmark-variable setting aligned
with the staged configuration. The cache loader can serve some requests that
are strict subsets of staged coverage, but that is a separate scientific run.
Surface variables remain unsupported by this staged-cache format.

## Documentation index

- [Google Cloud Batch deployment](google-cloud-batch-deployment.md): detailed
  architecture, resource, security, data, and failure-semantics reference.
- [Current code outline](code_outline.md): the broader calculation pipeline and
  module reference.
- [Eulerian Heat Budget - Reformulation](<Eulerian Heat Budget - Reformulation.md>):
  physics derivation and sign conventions.
