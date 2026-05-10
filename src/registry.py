"""
src/registry.py
Handles model versioning and registration in Vertex AI Model Registry.
Run directly to register a new model version:
    python -m src.registry --version v1.0.0 --experiment exp1 --promote
"""
import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from google.cloud import aiplatform, storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ID  = os.getenv("GCP_PROJECT_ID",  "predictive-maintenance-scania")
REGION      = os.getenv("GCP_REGION",      "europe-north1")
BUCKET      = os.getenv("GCS_BUCKET",      "predictive-maintenance-scania-artifacts")
MODEL_NAME  = "scania-component-x-failure-predictor"

EXPERIMENT_DISPLAY = {
    "exp1": "Log-Loss Baseline",
    "exp2": "Focal Loss",
    "exp3": "Cost-Aware Focal Loss (Proposed)",
}


def init_vertex():
    """Initialise Vertex AI SDK."""
    aiplatform.init(project=PROJECT_ID, location=REGION)
    logger.info(f"Vertex AI initialised — project={PROJECT_ID}, region={REGION}")


def upload_model_to_registry(
    version: str,
    experiment: str,
    metrics: dict = None,
    promote: bool = False,
) -> aiplatform.Model:
    """
    Register model in Vertex AI as a custom container model.
    No artifact_uri is used — avoids Vertex AI's pre-built
    container file validation entirely.
    Model artifacts live in GCS and are pulled at container startup.
    """
    init_vertex()

    display_name = f"{MODEL_NAME}-{experiment}-{version}"

    logger.info(f"Registering model: {display_name}")

    # Build labels
    labels = {
        "version":    version.replace(".", "-"),
        "experiment": experiment,
        "registered": datetime.utcnow().strftime("%Y%m%d"),
    }
    if metrics:
        for k, v in metrics.items():
            label_key = k.lower().replace(" ", "_")[:63]
            label_val = str(round(v, 4)).replace(".", "_")[:63]
            labels[label_key] = label_val

    # Custom serving container image
    custom_image_uri = (
        f"{REGION}-docker.pkg.dev/{PROJECT_ID}/"
        f"predictive-maintenance-scania-images/"
        f"scania-predictive-ml-api:latest"
    )

    # Check if Docker image exists yet
    try:
        import subprocess
        result = subprocess.run(
            ["gcloud", "artifacts", "docker", "images", "describe", custom_image_uri],
            capture_output=True, text=True
        )
        image_exists = result.returncode == 0
    except Exception:
        image_exists = False

    if not image_exists:
        logger.warning(
            "Docker image not yet in Artifact Registry. "
            "Skipping Vertex AI registration for now. "
            "Re-run this script after the first pipeline deploy."
        )
        # Write metadata to GCS and promote without Vertex AI registration
        _write_metadata_to_gcs(version, experiment, metrics or {})
        if promote:
            # Create a dummy model object reference for promote_model
            _write_promotion_record_directly(version, experiment)
        return None

    # Register as fully custom container model — no artifact_uri
    # so Vertex AI never validates model file names
    model = aiplatform.Model.upload(
        display_name=display_name,
        serving_container_image_uri=custom_image_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
        serving_container_environment_variables={
            "ACTIVE_MODEL":   experiment,
            "MODEL_VERSION":  version,
            "GCS_BUCKET":     BUCKET,
            "GCP_PROJECT_ID": PROJECT_ID,
            "OUTPUTS_DIR":    "/app/outputs/models",
        },
        description=(
            f"SCANIA Component X — "
            f"{EXPERIMENT_DISPLAY.get(experiment, experiment)} — {version}. "
            f"Artifacts at gs://{BUCKET}/models/{version}/"
        ),
        labels=labels,
    )

    logger.info(f"Model registered: {model.resource_name}")
    _write_metadata_to_gcs(version, experiment, metrics or {})

    if promote:
        promote_model(model, experiment, version)

    return model


def _write_promotion_record_directly(version: str, experiment: str):
    """
    Write promotion record directly to GCS when Vertex AI
    registration is skipped (Docker image not yet available).
    """
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)

    promotion_record = {
        "promoted_version": version,
        "experiment":       experiment,
        "model_resource":   "pending-first-pipeline-run",
        "promoted_at":      datetime.utcnow().isoformat(),
        "promoted_by":      os.getenv("GITHUB_ACTOR", "manual"),
    }

    blob = bucket.blob("models/promoted/current.json")
    blob.upload_from_string(
        json.dumps(promotion_record, indent=2),
        content_type="application/json",
    )
    logger.info(f"Promotion record written to gs://{BUCKET}/models/promoted/current.json")
    logger.info(json.dumps(promotion_record, indent=2))


def _write_metadata_to_gcs(version: str, experiment: str, metrics: dict):
    """Write model metadata to GCS for tracking."""
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)

    metadata = {
        "version":    version,
        "experiment": experiment,
        "metrics":    metrics,
        "artifact_uri": f"gs://{BUCKET}/models/{version}/",
        "files": {
            "pipeline": f"models/{version}/pipeline_state.pkl",
            "exp1":     f"models/{version}/exp1_log_loss.pkl",
            "exp2":     f"models/{version}/exp2_focal_loss.pkl",
            "exp3":     f"models/{version}/exp3_cost_aware_focal.pkl",
        },
        "registered_at": datetime.utcnow().isoformat(),
    }

    blob = bucket.blob(f"models/{version}/metadata.json")
    blob.upload_from_string(
        json.dumps(metadata, indent=2),
        content_type="application/json",
    )
    logger.info(f"Metadata written to gs://{BUCKET}/models/{version}/metadata.json")


def _write_metadata_to_gcs(version: str, experiment: str, metrics: dict):
    """
    Write a metadata JSON file to the model directory in GCS.
    This documents the model version without conflicting with
    Vertex AI's pre-built container file expectations.
    """
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)

    metadata = {
        "version":    version,
        "experiment": experiment,
        "metrics":    metrics,
        "files": {
            "pipeline":  f"models/{version}/pipeline_state.pkl",
            "exp1":      f"models/{version}/exp1_log_loss.pkl",
            "exp2":      f"models/{version}/exp2_focal_loss.pkl",
            "exp3":      f"models/{version}/exp3_cost_aware_focal.pkl",
        },
        "registered_at": datetime.utcnow().isoformat(),
    }

    blob = bucket.blob(f"models/{version}/metadata.json")
    blob.upload_from_string(
        json.dumps(metadata, indent=2),
        content_type="application/json",
    )
    logger.info(f"Metadata written to gs://{BUCKET}/models/{version}/metadata.json")


def promote_model(model: aiplatform.Model, experiment: str, version: str):
    """
    Write a 'promoted' marker to Cloud Storage so the deployment
    pipeline knows which version is production-ready.
    """
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)

    promotion_record = {
        "promoted_version": version,
        "experiment":       experiment,
        "model_resource":   model.resource_name,
        "promoted_at":      datetime.utcnow().isoformat(),
        "promoted_by":      os.getenv("GITHUB_ACTOR", "manual"),
    }

    blob = bucket.blob("models/promoted/current.json")
    blob.upload_from_string(
        json.dumps(promotion_record, indent=2),
        content_type="application/json",
    )
    logger.info(f"Promoted model written to gs://{BUCKET}/models/promoted/current.json")
    logger.info(json.dumps(promotion_record, indent=2))


def get_promoted_version() -> dict:
    """
    Read the currently promoted model version from Cloud Storage.
    Used by cloudbuild.yaml to know which version to deploy.
    """
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)
    blob   = bucket.blob("models/promoted/current.json")

    if not blob.exists():
        logger.warning("No promoted version found — using defaults")
        return {"promoted_version": "v1.0.0", "experiment": "exp1"}

    return json.loads(blob.download_as_text())


def list_registered_versions():
    """List all registered model versions in Vertex AI."""
    init_vertex()
    models = aiplatform.Model.list(
        filter=f'display_name="{MODEL_NAME}"',
        order_by="create_time desc",
    )
    logger.info(f"Found {len(models)} registered versions:")
    for m in models:
        logger.info(f"  {m.display_name} — {m.resource_name}")
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCANIA Model Registry CLI")
    parser.add_argument("--version",    required=True, help="Model version e.g. v1.0.0")
    parser.add_argument("--experiment", required=True, choices=["exp1", "exp2", "exp3"])
    parser.add_argument("--promote",    action="store_true", help="Promote this version to production")
    parser.add_argument("--list",       action="store_true", help="List all registered versions")
    parser.add_argument("--total-cost", type=float, help="Total cost metric from evaluation")
    parser.add_argument("--auc",        type=float, help="AUC-ROC metric from evaluation")
    args = parser.parse_args()

    if args.list:
        list_registered_versions()
    else:
        metrics = {}
        if args.total_cost: metrics["total_cost"] = args.total_cost
        if args.auc:        metrics["auc_roc"]    = args.auc

        upload_model_to_registry(
            version    = args.version,
            experiment = args.experiment,
            metrics    = metrics,
            promote    = args.promote,
        )