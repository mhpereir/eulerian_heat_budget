#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ID="eulerian-heat-budget"
REGION="us-central1"
BUCKET="eulerian-heat-budget-arco-staged-us-central1"
SERVICE_ACCOUNT_NAME="ehb-batch-retrieval"
CAMPAIGN_FILE=""
IMAGE_URI=""
JOB_ID=""
PARALLELISM=5
BOOT_DISK_GB=40

usage() {
  echo "Usage: $0 --campaign FILE --image-uri URI@sha256:DIGEST [--job-id ID] [--project ID] [--region REGION] [--bucket NAME] [--service-account NAME] [--parallelism N] [--boot-disk-gb N]" >&2
}

while (($#)); do
  case "$1" in
    --campaign) CAMPAIGN_FILE="$2"; shift 2 ;;
    --image-uri) IMAGE_URI="$2"; shift 2 ;;
    --job-id) JOB_ID="$2"; shift 2 ;;
    --project) PROJECT_ID="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --service-account) SERVICE_ACCOUNT_NAME="$2"; shift 2 ;;
    --parallelism) PARALLELISM="$2"; shift 2 ;;
    --boot-disk-gb) BOOT_DISK_GB="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "${CAMPAIGN_FILE}" && -n "${IMAGE_URI}" ]] || { usage; exit 2; }
command -v gcloud >/dev/null || { echo "gcloud is required." >&2; exit 2; }
command -v python3 >/dev/null || { echo "python3 is required." >&2; exit 2; }

PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd "${PROJECT_ROOT}"
CAMPAIGN_ID=$(python3 -m deployment.gcp.campaign "${CAMPAIGN_FILE}" --print-campaign-id)
if [[ -z "${JOB_ID}" ]]; then
  JOB_ID="${CAMPAIGN_ID:0:47}-$(date -u +%Y%m%d-%H%M%S)"
fi
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "${TEMP_DIR}"' EXIT
NORMALIZED_CAMPAIGN="${TEMP_DIR}/campaign.json"
JOB_CONFIG="${TEMP_DIR}/batch-job.json"

python3 -m deployment.gcp.render_batch_job \
  --campaign "${CAMPAIGN_FILE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --bucket "${BUCKET}" \
  --image-uri "${IMAGE_URI}" \
  --service-account-email "${SERVICE_ACCOUNT_EMAIL}" \
  --job-id "${JOB_ID}" \
  --parallelism "${PARALLELISM}" \
  --boot-disk-gb "${BOOT_DISK_GB}" \
  --output "${JOB_CONFIG}" \
  --normalized-campaign-output "${NORMALIZED_CAMPAIGN}"

CAMPAIGN_URI="gs://${BUCKET}/campaigns/${CAMPAIGN_ID}/campaign.json"
EXISTING_CAMPAIGN="${TEMP_DIR}/existing-campaign.json"
compare_existing_campaign() {
  gcloud storage cp "${CAMPAIGN_URI}" "${EXISTING_CAMPAIGN}" --project "${PROJECT_ID}" >/dev/null
  python3 -m deployment.gcp.campaign "${EXISTING_CAMPAIGN}" --normalize-to "${TEMP_DIR}/existing-normalized.json"
  if ! cmp -s "${NORMALIZED_CAMPAIGN}" "${TEMP_DIR}/existing-normalized.json"; then
    echo "Campaign ${CAMPAIGN_ID} already exists with a different configuration." >&2
    return 1
  fi
}

if gcloud storage objects describe "${CAMPAIGN_URI}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  compare_existing_campaign
else
  if ! gcloud storage cp \
    "${NORMALIZED_CAMPAIGN}" "${CAMPAIGN_URI}" \
    --project "${PROJECT_ID}" \
    --if-generation-match=0; then
    echo "[info] Campaign was created concurrently; validating the stored configuration"
    compare_existing_campaign
  fi
fi

JOB_CONFIG_URI="gs://${BUCKET}/campaigns/${CAMPAIGN_ID}/batch-jobs/${JOB_ID}.json"
gcloud storage cp "${JOB_CONFIG}" "${JOB_CONFIG_URI}" --project "${PROJECT_ID}"

echo "[info] Submitting Batch job ${JOB_ID}"
gcloud batch jobs submit "${JOB_ID}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --config "${JOB_CONFIG}"

echo "[info] Job submitted: projects/${PROJECT_ID}/locations/${REGION}/jobs/${JOB_ID}"
echo "[info] Campaign output: gs://${BUCKET}/campaigns/${CAMPAIGN_ID}/"
