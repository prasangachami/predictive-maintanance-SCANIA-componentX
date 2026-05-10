"""
src/data_loader.py
Loads raw data from either local disk or GCS depending on environment.
Usage:
    from src.data_loader import get_data_dir, download_data_from_gcs
"""
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

GCS_BUCKET  = os.getenv("GCS_BUCKET", "predictive-maintenance-scania-artifacts")
GCS_DATA    = f"gs://{GCS_BUCKET}/data/raw"
LOCAL_DATA  = os.getenv("DATA_DIR", "data/raw")


def get_data_dir() -> str:
    """
    Returns the correct data directory path.
    - In CI/CD or Cloud Run: downloads from GCS first, returns local path
    - Locally: returns local data/raw/ path directly
    """
    if os.getenv("CI") or os.getenv("CLOUD_RUN_JOB"):
        download_data_from_gcs()
    return LOCAL_DATA


def download_data_from_gcs(local_dir: str = LOCAL_DATA):
    """Download raw data files from GCS to local disk."""
    try:
        from google.cloud import storage
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        client  = storage.Client()
        bucket  = client.bucket(GCS_BUCKET)
        blobs   = bucket.list_blobs(prefix="data/raw/")

        for blob in blobs:
            filename = blob.name.split("/")[-1]
            if not filename.endswith(".csv"):
                continue
            dest = os.path.join(local_dir, filename)
            if not os.path.exists(dest):
                logger.info(f"Downloading {blob.name} → {dest}")
                blob.download_to_filename(dest)
            else:
                logger.info(f"Already exists, skipping: {dest}")

        logger.info("All data files ready.")

    except Exception as e:
        logger.error(f"Failed to download data from GCS: {e}")
        raise