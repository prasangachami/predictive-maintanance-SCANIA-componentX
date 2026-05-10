from pydantic import BaseModel, Field
from typing import Optional


class TruckInput(BaseModel):
    """
    A single truck's operational snapshot for inference.
    The preprocessing pipeline (pipeline_state.pkl) handles all
    feature engineering internally — you only need to pass the
    raw operational readout fields here.
    """
    truck_id: str = Field(..., description="Unique truck identifier")

    # Raw operational readout features (from operational_readouts.csv)
    # These are passed directly into pipeline_state.pkl for preprocessing
    readout_data: dict = Field(
        ...,
        description=(
            "Raw dict of operational readout columns for this truck. "
            "Keys must match train_operational_readouts.csv column names exactly."
        )
    )

    # Optional: truck specification fields (from specifications.csv)
    spec_data: Optional[dict] = Field(
        None,
        description="Optional truck specification fields (from specifications.csv)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "truck_id": "truck_001",
                "readout_data": {
                    "aa_000": 1523.4,
                    "ab_000": 872.1,
                    "ac_000": 304.9,
                },
                "spec_data": None
            }
        }


class PredictionResponse(BaseModel):
    truck_id: str

    # Binary prediction (from binary label system used in training)
    failure_predicted: bool = Field(..., description="True = failure within prediction window")
    failure_probability: float = Field(..., description="P(failure) from model, 0.0–1.0")

    # Cost-aware output (from 5-class temporal_class system)
    temporal_class: int = Field(
        ...,
        description=(
            "0 = healthy (>48 steps), "
            "1 = 12–24 steps, "
            "2 = 24–48 steps, "
            "3 = 6–12 steps, "
            "4 = imminent (0–6 steps)"
        )
    )
    urgency_label: str = Field(..., description="Human-readable urgency: HEALTHY / MONITOR / SOON / URGENT / IMMINENT")
    estimated_cost_risk: float = Field(..., description="Expected cost based on industrial cost matrix")

    # Which experiment model was used
    model_used: str = Field(..., description="exp1_log_loss | exp2_focal_loss | exp3_cost_aware_focal")
    model_version: str


class HealthResponse(BaseModel):
    status: str
    pipeline_loaded: bool
    models_loaded: dict   # {"exp1": True, "exp2": True, "exp3": True}
    active_model: str
    version: str