#!/usr/bin/env bash
# scripts/deploy_cloud_run.sh — Deploy CausalBound API & War Room UI to Google Cloud Run
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID before deploying}"
REGION="${GCP_REGION:-us-central1}"
API_SERVICE="causalbound-api"
UI_SERVICE="causalbound-war-room"

echo "1/3 Deploying FastAPI Neuro-Symbolic Control Plane to Cloud Run..."
gcloud run deploy "${API_SERVICE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --source . \
  --dockerfile Dockerfile.api \
  --cpu 1 \
  --memory 2Gi \
  --allow-unauthenticated \
  --set-env-vars "CAUSALBOUND_FAST_MODEL=${CAUSALBOUND_FAST_MODEL:-gemini-3.8-flash},CAUSALBOUND_PRO_MODEL=${CAUSALBOUND_PRO_MODEL:-gemini-3.1-pro}"

API_URL="$(gcloud run services describe "${API_SERVICE}" --project "${PROJECT_ID}" --region "${REGION}" --format 'value(status.url)')"
echo "API live at: ${API_URL}"

echo "2/3 Deploying Next.js 15 Risk War Room UI to Cloud Run..."
gcloud run deploy "${UI_SERVICE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --source ./frontend \
  --cpu 1 \
  --memory 1Gi \
  --allow-unauthenticated \
  --set-build-env-vars "NEXT_PUBLIC_API_URL=${API_URL}"

UI_URL="$(gcloud run services describe "${UI_SERVICE}" --project "${PROJECT_ID}" --region "${REGION}" --format 'value(status.url)')"
echo "3/3 DEPLOYMENT COMPLETE"
echo "  API URL : ${API_URL}"
echo "  UI URL  : ${UI_URL}"
