import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import TruckInput, PredictionResponse, HealthResponse
from app.predict import load_all, run_inference, get_models_status, ACTIVE_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_raw_version = os.getenv("MODEL_VERSION", "1.0.0")
MODEL_VERSION = _raw_version if _raw_version and _raw_version.strip() else "1.0.0"
# Strip leading 'v' if present (FastAPI requires semver e.g. 1.0.0 not v1.0.0)
MODEL_VERSION = MODEL_VERSION.lstrip("v")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup — non-blocking so server starts even if models fail."""
    logger.info("Starting SCANIA Predictive Maintenance API...")
    try:
        load_all()
        logger.info("All models loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        logger.warning("API starting without models — /predict will return 503")
    except Exception as e:
        logger.error(f"Unexpected error loading models: {e}")
        logger.warning("API starting in degraded mode")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SCANIA Component X — Predictive Maintenance API",
    description=(
        "Cost-aware failure prediction for SCANIA truck Component X. "
        "Uses LightGBM with cost-aware focal loss."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    try:
        models_status = get_models_status()
        all_ok = any(models_status.values())
    except Exception:
        models_status = {"exp1": False, "exp2": False, "exp3": False}
        all_ok = False

    return HealthResponse(
        status="ok" if all_ok else "degraded",
        pipeline_loaded=all_ok,
        models_loaded=models_status,
        active_model=ACTIVE_MODEL,
        version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(payload: TruckInput):
    try:
        result = run_inference(
            truck_id=payload.truck_id,
            readout_data=payload.readout_data,
            spec_data=payload.spec_data,
        )
        return PredictionResponse(truck_id=payload.truck_id, **result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error for {payload.truck_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "SCANIA Predictive Maintenance API",
        "active_model": ACTIVE_MODEL,
        "version": MODEL_VERSION,
        "docs": "/docs",
    }