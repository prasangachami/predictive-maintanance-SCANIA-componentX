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

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting — loading pipeline and models...")
    load_all()
    logger.info("All models ready.")
    yield


app = FastAPI(
    title="SCANIA Component X — Predictive Maintenance API",
    description=(
        "Cost-aware failure prediction for SCANIA truck Component X. "
        "Uses LightGBM with cost-aware focal loss trained on temporal failure classes."
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
    models_status = get_models_status()
    all_ok = all(models_status.values())
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        pipeline_loaded=True,
        models_loaded=models_status,
        active_model=ACTIVE_MODEL,
        version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(payload: TruckInput):
    """
    Predict Component X failure for a truck.

    - Runs raw features through `pipeline_state.pkl` preprocessing
    - Applies the active experiment model (default: exp3 cost-aware focal)
    - Returns binary prediction + temporal urgency class + cost risk estimate
    """
    try:
        result = run_inference(
            truck_id=payload.truck_id,
            readout_data=payload.readout_data,
            spec_data=payload.spec_data,
        )
        return PredictionResponse(truck_id=payload.truck_id, **result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error for {payload.truck_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "SCANIA Predictive Maintenance API",
        "active_model": ACTIVE_MODEL,
        "docs": "/docs",
    }