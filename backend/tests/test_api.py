import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="module")
def client():
    mock_pipeline = MagicMock()
    mock_pipeline.transform.return_value = np.zeros((1, 10))

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.2, 0.8]]
    mock_model.optimal_threshold_ = 0.5

    with patch("app.predict._pipeline", mock_pipeline), \
         patch("app.predict._models", {
             "exp1": mock_model,
             "exp2": mock_model,
             "exp3": mock_model,
         }):
        from app.main import app
        yield TestClient(app)


SAMPLE_PAYLOAD = {
    "truck_id": "truck_001",
    "readout_data": {
        "aa_000": 1523.4,
        "ab_000": 872.1,
        "ac_000": 304.9,
    },
    "spec_data": None,
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["pipeline_loaded"] is True
    assert "models_loaded" in data
    assert "active_model" in data


def test_predict_returns_all_fields(client):
    r = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert data["truck_id"] == "truck_001"
    assert isinstance(data["failure_predicted"], bool)
    assert 0.0 <= data["failure_probability"] <= 1.0
    assert data["temporal_class"] in [0, 1, 2, 3, 4]
    assert data["urgency_label"] in ["HEALTHY", "MONITOR", "SOON", "URGENT", "IMMINENT"]
    assert data["estimated_cost_risk"] >= 0.0
    assert data["model_used"] in ["exp1", "exp2", "exp3"]


def test_predict_missing_truck_id(client):
    bad_payload = {"readout_data": {"aa_000": 1.0}}
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_predict_missing_readout_data(client):
    bad_payload = {"truck_id": "truck_001"}
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "active_model" in r.json()