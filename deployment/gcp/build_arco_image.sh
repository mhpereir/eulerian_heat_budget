#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ID="eulerian-heat-budget"
REGION="us-central1"
REPOSITORY="ehb-batch"
IMAGE_NAME="arco-retrieval"
TAG=""
ALLOW_DIRTY=0

usage() {
  echo "Usage: $0 [--project ID] [--region REGION] [--repository NAME] [--image NAME] [--tag TAG] [--allow-dirty]" >&2
}

while (($#)); do
  case "$1" in
    --project) PROJECT_ID="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --repository) REPOSITORY="$2"; shift 2 ;;
    --image) IMAGE_NAME="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

command -v gcloud >/dev/null || { echo "gcloud is required." >&2; exit 2; }
PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd "${PROJECT_ROOT}"

if [[ "${ALLOW_DIRTY}" != "1" ]] && [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to build a production image from a dirty worktree. Commit the changes or pass --allow-dirty." >&2
  exit 2
fi

if [[ -z "${TAG}" ]]; then
  TAG="git-$(git rev-parse --short=12 HEAD)"
fi
IMAGE_BASE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}"
TAGGED_IMAGE="${IMAGE_BASE}:${TAG}"

echo "[info] Building ${TAGGED_IMAGE} with Cloud Build"
gcloud builds submit \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --tag "${TAGGED_IMAGE}" \
  .

DIGEST=$(gcloud artifacts docker images describe \
  "${TAGGED_IMAGE}" \
  --project "${PROJECT_ID}" \
  --format='value(image_summary.digest)')
if [[ ! "${DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Could not resolve an immutable digest for ${TAGGED_IMAGE}; got ${DIGEST@Q}." >&2
  exit 1
fi

echo "IMAGE_URI=${IMAGE_BASE}@${DIGEST}"
