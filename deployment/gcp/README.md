# Google Cloud Batch ARCO retrieval

This deployment runs one campaign year per Google Cloud Batch task. Each task
writes its SQLite and Zarr cache to local scratch space and only publishes a
completed, verified yearly shard to Cloud Storage. The bucket is not used as a
live cache filesystem.

## 1. Prepare the cloud resources

Authenticate `gcloud`, select the billing project, and run:

```bash
./deployment/gcp/setup_batch_resources.sh
```

The defaults create these resources in project `eulerian-heat-budget`:

- Bucket `eulerian-heat-budget-arco-staged-us-central1`
- Artifact Registry repository `ehb-batch` in `us-central1`
- Runtime service account `ehb-batch-retrieval`

The active `gcloud` user receives Batch Job Editor, Service Account User, and
Storage Object User on the campaign bucket. Pass
`--submitter user:name@example.com` to grant those permissions to a different
principal. All resource names can be overridden with command-line options.
The setup script is safe to rerun and fails if an existing bucket is in the
wrong region or an existing Artifact Registry repository is not a Docker
repository.

## 2. Define a campaign

Copy `deployment/gcp/campaign.example.json` and replace the campaign ID and all
scientific parameters. Submission never falls back to the PBS production
settings.

The required document shape is:

```json
{
  "campaign_id": "unique-lowercase-id",
  "years": {"start": 1940, "end": 2025},
  "season": {"start_month_day": "05-01", "end_month_day": "10-31"},
  "domain": {
    "region": "eastern_canada",
    "margin_n": 1,
    "zg_top_pa": 70000,
    "zg_bottom": "surface_pressure",
    "allow_bottom_overflow": true
  },
  "staging": {
    "time_chunk": "month",
    "attempt_timeout_seconds": 10800,
    "include_benchmark_variables": false
  }
}
```

Use `domain.bbox` instead of `domain.region` for explicit bounds. A
pressure-level bottom also requires `domain.zg_bottom_pa`. Validate and
normalize a file without submitting it:

```bash
python3 -m deployment.gcp.campaign my-campaign.json --normalize-to /tmp/campaign.json
```

For the first canary, use a unique campaign ID and set the start and end year
to the same year.

## 3. Build the image

Commit the deployment changes so the image has reproducible source provenance,
then run:

```bash
./deployment/gcp/build_arco_image.sh
```

Cloud Build installs `requirements-arco.txt` with hash verification, performs a
real local Zarr write/read smoke test, and runs the complete pytest suite. The
last line reports an immutable value such as:

```text
IMAGE_URI=us-central1-docker.pkg.dev/eulerian-heat-budget/ehb-batch/arco-retrieval@sha256:...
```

Use that digest URI for submission. Tags such as `latest` are rejected.

## 4. Submit and monitor a canary

```bash
./deployment/gcp/submit_arco_batch.sh \
  --campaign my-canary.json \
  --image-uri "${IMAGE_URI}" \
  --parallelism 1
```

The submission command validates the campaign, records its normalized JSON and
rendered job JSON in the bucket, then submits the Batch job. A campaign ID
cannot be reused with different parameters.

Inspect job state and task logs with:

```bash
gcloud batch jobs describe JOB_ID --location us-central1
gcloud logging read \
  'labels."goog-batch-job-id"="JOB_ID"' \
  --project eulerian-heat-budget \
  --limit 100
```

A completed year has this layout:

```text
campaigns/CAMPAIGN_ID/
  campaign.json
  batch-jobs/JOB_ID.json
  shards/year=YYYY/
    cache.sqlite
    tiles/*.zarr/
    shard-manifest.json
    _SUCCESS.json
```

`_SUCCESS.json` is written only after the published path and size inventory has
been checked. A retry removes an incomplete yearly prefix, while a valid
completed shard is skipped.

## 5. Download and consolidate

Download the campaign onto a POSIX filesystem:

```bash
mkdir -p data/CAMPAIGN_ID
gcloud storage rsync --recursive \
  gs://eulerian-heat-budget-arco-staged-us-central1/campaigns/CAMPAIGN_ID \
  data/CAMPAIGN_ID
```

Run the consolidator in the same locked image or a matching local environment:

```bash
docker run --rm \
  --entrypoint python \
  --mount type=bind,src="${PWD}/data/CAMPAIGN_ID",dst=/data \
  "${IMAGE_URI}" \
  scripts/consolidate_staged_arco_cache.py --cache-root /data
```

The command verifies all object checksums and hourly coverage before atomically
creating `data/CAMPAIGN_ID/cache.sqlite`. Its catalog paths point to the nested
yearly Zarr stores, so the existing budget command can use that directory as
`--staged-cache-root` without loader changes.

Optionally upload the generated root metadata so later downloads already
contain the combined catalog:

```bash
gcloud storage cp \
  data/CAMPAIGN_ID/cache.sqlite \
  data/CAMPAIGN_ID/consolidation.json \
  gs://eulerian-heat-budget-arco-staged-us-central1/campaigns/CAMPAIGN_ID/
```

After the one-year canary has downloaded, consolidated, and loaded
successfully, submit the full campaign with the default parallelism of five.
