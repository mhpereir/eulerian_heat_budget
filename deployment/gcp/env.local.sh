# Stable deployment configuration
export PROJECT_ID="eulerian-heat-budget"
export REGION="us-central1"
export BUCKET="eulerian-heat-budget-arco-staged-us-central1"
export REPOSITORY="ehb-batch"
export IMAGE_NAME="arco-retrieval"
export SERVICE_ACCOUNT_NAME="ehb-batch-retrieval"

# Operator and campaign configuration
export CAMPAIGN_CONFIG_DIR="${HOME}/work/google_cloud/ehb-campaigns"
export CAMPAIGN_FILE="${CAMPAIGN_CONFIG_DIR}/gulf-usa-1940-2025-surface-700hpa.json"
export CANARY_FILE="${CAMPAIGN_CONFIG_DIR}/gulf-usa-canary-2025-surface-700hpa.json"

export CAMPAIGN_ID="$(python3 -m deployment.gcp.campaign \
  "${CAMPAIGN_FILE}" \
  --print-campaign-id)"

export CANARY_ID="$(python3 -m deployment.gcp.campaign \
  "${CANARY_FILE}" \
  --print-campaign-id)"

## How to recover IMAGE_URI from previous log:
# export BUILD_LOG="/tmp/ehb-cloud-build-20260716-172507.log"
# export IMAGE_URI="$(sed -n 's/^IMAGE_URI=//p' "${BUILD_LOG}" | tail -n 1)"
# printf '%s\n' "${IMAGE_URI}"
