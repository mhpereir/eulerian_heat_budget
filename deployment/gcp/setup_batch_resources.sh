#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ID="eulerian-heat-budget"
REGION="us-central1"
BUCKET="eulerian-heat-budget-arco-staged-us-central1"
REPOSITORY="ehb-batch"
SERVICE_ACCOUNT_NAME="ehb-batch-retrieval"
SUBMITTER=""

usage() {
  echo "Usage: $0 [--project ID] [--region REGION] [--bucket NAME] [--repository NAME] [--service-account NAME] [--submitter IAM_PRINCIPAL]" >&2
}

while (($#)); do
  case "$1" in
    --project) PROJECT_ID="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --repository) REPOSITORY="$2"; shift 2 ;;
    --service-account) SERVICE_ACCOUNT_NAME="$2"; shift 2 ;;
    --submitter) SUBMITTER="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

command -v gcloud >/dev/null || { echo "gcloud is required." >&2; exit 2; }
if [[ -z "${SUBMITTER}" ]]; then
  ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format='value(account)' | head -n 1)
  if [[ -z "${ACTIVE_ACCOUNT}" ]]; then
    echo "No active gcloud account. Authenticate or pass --submitter explicitly." >&2
    exit 2
  fi
  SUBMITTER="user:${ACTIVE_ACCOUNT}"
fi

SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
RUNTIME_MEMBER="serviceAccount:${SERVICE_ACCOUNT_EMAIL}"

echo "[info] Enabling Google Cloud APIs"
gcloud services enable \
  batch.googleapis.com \
  compute.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  logging.googleapis.com \
  storage.googleapis.com \
  --project "${PROJECT_ID}"

if ! gcloud storage buckets describe "gs://${BUCKET}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  echo "[info] Creating gs://${BUCKET}"
  gcloud storage buckets create "gs://${BUCKET}" \
    --project "${PROJECT_ID}" \
    --location "${REGION}" \
    --default-storage-class STANDARD \
    --uniform-bucket-level-access \
    --public-access-prevention \
    --soft-delete-duration 7d
fi
BUCKET_LOCATION=$(gcloud storage buckets describe "gs://${BUCKET}" \
  --project "${PROJECT_ID}" --format='value(location)')
if [[ "${BUCKET_LOCATION,,}" != "${REGION,,}" ]]; then
  echo "Existing bucket gs://${BUCKET} is in ${BUCKET_LOCATION}, expected ${REGION}." >&2
  exit 1
fi
gcloud storage buckets update "gs://${BUCKET}" \
  --project "${PROJECT_ID}" \
  --default-storage-class STANDARD \
  --uniform-bucket-level-access \
  --public-access-prevention \
  --soft-delete-duration 7d

if ! gcloud artifacts repositories describe "${REPOSITORY}" \
  --project "${PROJECT_ID}" --location "${REGION}" >/dev/null 2>&1; then
  echo "[info] Creating Artifact Registry repository ${REPOSITORY}"
  gcloud artifacts repositories create "${REPOSITORY}" \
    --project "${PROJECT_ID}" \
    --location "${REGION}" \
    --repository-format docker \
    --description "Eulerian heat budget Batch container images"
fi
REPOSITORY_FORMAT=$(gcloud artifacts repositories describe "${REPOSITORY}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --format='value(format)')
if [[ "${REPOSITORY_FORMAT}" != "DOCKER" ]]; then
  echo "Existing Artifact Registry repository ${REPOSITORY} has format ${REPOSITORY_FORMAT}, expected DOCKER." >&2
  exit 1
fi

if ! gcloud iam service-accounts describe "${SERVICE_ACCOUNT_EMAIL}" \
  --project "${PROJECT_ID}" >/dev/null 2>&1; then
  echo "[info] Creating service account ${SERVICE_ACCOUNT_EMAIL}"
  gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
    --project "${PROJECT_ID}" \
    --display-name "EHB Batch ARCO retrieval"
fi

for role in roles/batch.agentReporter roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "${RUNTIME_MEMBER}" \
    --role "${role}" \
    --condition=None >/dev/null
done
gcloud artifacts repositories add-iam-policy-binding "${REPOSITORY}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --member "${RUNTIME_MEMBER}" \
  --role roles/artifactregistry.reader >/dev/null
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --project "${PROJECT_ID}" \
  --member "${RUNTIME_MEMBER}" \
  --role roles/storage.objectUser >/dev/null
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --project "${PROJECT_ID}" \
  --member "${SUBMITTER}" \
  --role roles/storage.objectUser >/dev/null

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member "${SUBMITTER}" \
  --role roles/batch.jobsEditor \
  --condition=None >/dev/null
gcloud iam service-accounts add-iam-policy-binding "${SERVICE_ACCOUNT_EMAIL}" \
  --project "${PROJECT_ID}" \
  --member "${SUBMITTER}" \
  --role roles/iam.serviceAccountUser >/dev/null

echo "[info] Batch resources are ready"
echo "BUCKET=${BUCKET}"
echo "ARTIFACT_REPOSITORY=${REPOSITORY}"
echo "SERVICE_ACCOUNT_EMAIL=${SERVICE_ACCOUNT_EMAIL}"
