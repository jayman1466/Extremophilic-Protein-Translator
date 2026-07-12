#!/usr/bin/env bash
# Deploy the design portal to Cloud Run (run from your machine — gcloud must be
# authenticated and pointed at the target project). Creates a GCS bucket for the
# SQLite DB + result files and mounts it into the service via GCS FUSE.
set -euo pipefail

PROJECT="${PROJECT:-extremolith}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-ept-portal}"
BUCKET="${BUCKET:-${PROJECT}-ept-portal-data}"   # must be globally unique
MOUNT="/mnt/gcs"

echo ">> project=$PROJECT region=$REGION service=$SERVICE bucket=gs://$BUCKET"
gcloud config set project "$PROJECT"

# 1. enable required APIs (idempotent)
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com storage.googleapis.com

# 2. create the storage bucket if absent
if ! gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://$BUCKET" --location="$REGION" --uniform-bucket-level-access
  echo ">> created gs://$BUCKET"
else
  echo ">> bucket gs://$BUCKET already exists"
fi

# 3. deploy from source (Cloud Build builds the Dockerfile in this dir).
#    GCS FUSE volume mount requires Cloud Run gen2 execution environment.
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --execution-environment gen2 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --set-env-vars "DATA_ROOT=$MOUNT,DB_PATH=$MOUNT/jobs.db" \
  --add-volume "name=gcsvol,type=cloud-storage,bucket=$BUCKET" \
  --add-volume-mount "volume=gcsvol,mount-path=$MOUNT"

echo ">> deployed. URL:"
gcloud run services describe "$SERVICE" --region "$REGION" --format="value(status.url)"
