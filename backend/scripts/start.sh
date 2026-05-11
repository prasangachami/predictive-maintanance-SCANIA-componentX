#!/bin/bash
set -e

echo "=== SCANIA ML API Startup ==="
echo "Project:  $GCP_PROJECT_ID"
echo "Bucket:   $GCS_BUCKET"
echo "Model:    $ACTIVE_MODEL"
echo "Version:  $MODEL_VERSION"

# Create model directory
mkdir -p outputs/models

# Download promoted version marker
echo "Fetching promoted model version..."
gcloud storage cp \
    gs://$GCS_BUCKET/models/promoted/current.json \
    /tmp/promoted.json 2>/dev/null || echo "No promotion marker found, using defaults"

# Parse promoted version if marker exists
if [ -f /tmp/promoted.json ]; then
    PROMOTED_VERSION=$(python3 -c "import json; d=json.load(open('/tmp/promoted.json')); print(d['promoted_version'])")
    PROMOTED_EXPERIMENT=$(python3 -c "import json; d=json.load(open('/tmp/promoted.json')); print(d['experiment'])")
    export MODEL_VERSION=$PROMOTED_VERSION
    export ACTIVE_MODEL=$PROMOTED_EXPERIMENT
    echo "Promoted version: $MODEL_VERSION ($ACTIVE_MODEL)"
fi

# Download model artifacts from GCS
echo "Downloading model artifacts..."
gcloud storage cp -r \
    gs://$GCS_BUCKET/models/$MODEL_VERSION/ \
    outputs/models/

echo "Downloaded files:"
ls -la outputs/models/

# Start FastAPI server
echo "Starting API server on port $PORT..."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1