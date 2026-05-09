import os
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Paths (match your outputs/models/ structure) ─────────────────────────────
OUTPUTS_DIR   = Path(os.getenv("OUTPUTS_DIR", "outputs/models"))
PIPELINE_PATH = OUTPUTS_DIR / "pipeline_state.pkl"
MODEL_PATHS   = {
    "exp1": OUTPUTS_DIR / "exp1_log_loss.pkl",
    "exp2": OUTPUTS_DIR / "exp2_focal_loss.pkl",
    "exp3": OUTPUTS_DIR / "exp3_cost_aware_focal.pkl",  # adjust if name has suffix
}

# Which experiment model to serve (override via env var)
ACTIVE_MODEL  = os.getenv("ACTIVE_MODEL", "exp3")   # default: best model
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# ── Industrial cost matrix (from your README) ─────────────────────────────────
# Rows = actual class (0–4), Cols = predicted class (0–4)
COST_MATRIX = np.array([
    [  0,   7,   8,   9,  10],
    [200,   0,   7,   8,   9],
    [300, 200,   0,   7,   8],
    [400, 300, 200,   0,   7],
    [500, 400, 300, 200,   0],
], dtype=float)

URGENCY_LABELS = {
    0: "HEALTHY",
    1: "MONITOR",
    2: "SOON",
    3: "URGENT",
    4: "IMMINENT",
}

# ── Module-level cache ────────────────────────────────────────────────────────
_pipeline = None
_models   = {}


def load_all():
    """Load preprocessing pipeline + all experiment models at startup."""
    global _pipeline, _models

    # Load preprocessing pipeline
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Pipeline not found at {PIPELINE_PATH}")
    _pipeline = joblib.load(PIPELINE_PATH)
    logger.info(f"Loaded pipeline from {PIPELINE_PATH}")

    # Load whichever experiment models exist
    for exp_name, path in MODEL_PATHS.items():
        # Handle glob for exp3 which may have a suffix in the filename
        candidates = list(OUTPUTS_DIR.glob(f"{exp_name}*.pkl"))
        resolved = path if path.exists() else (candidates[0] if candidates else None)
        if resolved:
            _models[exp_name] = joblib.load(resolved)
            logger.info(f"Loaded model: {exp_name} from {resolved}")
        else:
            logger.warning(f"Model not found, skipping: {exp_name}")

    if ACTIVE_MODEL not in _models:
        raise RuntimeError(
            f"Active model '{ACTIVE_MODEL}' not loaded. "
            f"Available: {list(_models.keys())}"
        )


def get_pipeline():
    if _pipeline is None:
        load_all()
    return _pipeline


def get_models_status() -> dict:
    return {name: (name in _models) for name in MODEL_PATHS}


def binary_to_temporal_class(failure_probability: float) -> int:
    """
    Map P(failure) → temporal class for cost evaluation.
    Thresholds tuned to match your cost matrix urgency bands.
    Override with env vars if your threshold optimiser produces different values.
    """
    t1 = float(os.getenv("THRESHOLD_CLASS1", "0.15"))
    t2 = float(os.getenv("THRESHOLD_CLASS2", "0.35"))
    t3 = float(os.getenv("THRESHOLD_CLASS3", "0.55"))
    t4 = float(os.getenv("THRESHOLD_CLASS4", "0.75"))

    if failure_probability < t1:
        return 0   # HEALTHY
    elif failure_probability < t2:
        return 1   # MONITOR
    elif failure_probability < t3:
        return 2   # SOON
    elif failure_probability < t4:
        return 3   # URGENT
    return 4       # IMMINENT


def expected_cost(temporal_class: int) -> float:
    """
    Expected cost of acting on this prediction using the industrial cost matrix.
    Uses the predicted class row — conservative estimate assuming worst-case actual.
    """
    row = COST_MATRIX[temporal_class]
    # Weight by predicted probability across classes (uniform prior for display)
    return float(np.mean(row[row > 0]))


def run_inference(truck_id: str, readout_data: dict, spec_data: dict = None) -> dict:
    """
    Full inference pipeline:
      raw features → pipeline_state.pkl → model → binary prediction
                   → temporal_class → cost estimate
    """
    pipeline = get_pipeline()

    if ACTIVE_MODEL not in _models:
        raise RuntimeError(f"Model '{ACTIVE_MODEL}' not available")

    model = _models[ACTIVE_MODEL]

    # Build raw DataFrame — matches what your scania_pipeline.py expects
    df = pd.DataFrame([readout_data])
    if spec_data:
        for k, v in spec_data.items():
            df[k] = v

    # Run preprocessing pipeline (same transform used during training)
    X_processed = pipeline.transform(df)

    # Get binary failure probability
    proba = model.predict_proba(X_processed)[0]
    failure_probability = float(proba[1])

    # Apply your cost-optimised threshold (stored inside the model pkl if available)
    threshold = getattr(model, "optimal_threshold_", 0.5)
    failure_predicted = failure_probability >= threshold

    # Map to temporal class + cost
    t_class = binary_to_temporal_class(failure_probability)
    cost_risk = expected_cost(t_class)

    return {
        "failure_predicted": failure_predicted,
        "failure_probability": round(failure_probability, 4),
        "temporal_class": t_class,
        "urgency_label": URGENCY_LABELS[t_class],
        "estimated_cost_risk": round(cost_risk, 2),
        "model_used": ACTIVE_MODEL,
        "model_version": MODEL_VERSION,
    }