"""Integration tests for the FastAPI prediction endpoint."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

SAMPLE_PAYLOAD = {
    "hour": 14102100,
    "banner_pos": 0,
    "site_id": "abc123",
    "site_domain": "domain1",
    "site_category": "arts",
    "app_id": "app1",
    "app_domain": "appdomain1",
    "app_category": "games",
    "device_id": "dev1",
    "device_ip": "1.2.3.4",
    "device_model": "modelA",
    "device_type": 1,
    "C1": 1005, "C14": 21689, "C15": 320, "C16": 50,
    "C17": 112, "C18": 0, "C19": 35, "C20": -1, "C21": 79,
}


@pytest.fixture
def client():
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])

    mock_store = MagicMock()
    mock_store.get.return_value = {}

    mock_pred_logger = MagicMock()
    mock_pred_logger.enabled = False

    with (
        patch("src.api.main.load_model_and_encoders", return_value=(mock_model, {}, "1")),
        patch("src.api.main.get_store", return_value=mock_store),
        patch("src.api.main.get_logger", return_value=mock_pred_logger),
    ):
        from src.api.main import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "1"


def test_predict_returns_probability(client):
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "click_probability" in data
    assert 0.0 <= data["click_probability"] <= 1.0
    assert data["model_version"] == "1"


def test_predict_will_click_threshold(client):
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = response.json()
    assert data["will_click"] == (data["click_probability"] >= data["threshold"])


def test_feedback_when_logger_disabled(client):
    response = client.post(
        "/feedback",
        json={"prediction_id": 1, "actual_click": 1},
    )
    assert response.status_code == 503
