"""FastAPI serving layer for ad-click prediction."""

import os
import pickle
import mlflow.lightgbm
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

app = FastAPI(title="Ad-Click Prediction API", version="0.1.0")

MODEL_NAME = os.getenv("MODEL_NAME", "ad-click-lgbm")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

_model = None
_encoders = None


def _load_model():
    global _model, _encoders
    mlflow.set_tracking_uri(MLFLOW_URI)
    logger.info(f"Loading model {MODEL_NAME} @ {MODEL_STAGE}")
    _model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if versions:
        run_id = versions[0].run_id
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="encoders.pkl"
        )
        with open(artifact_path, "rb") as f:
            _encoders = pickle.load(f)
    logger.info("Model and encoders loaded successfully")


@app.on_event("startup")
async def startup():
    _load_model()


class AdRequest(BaseModel):
    hour: int = Field(..., description="Hour in YYMMDDHH format, e.g. 14102100")
    banner_pos: int = 0
    site_id: str = "unknown"
    site_domain: str = "unknown"
    site_category: str = "unknown"
    app_id: str = "unknown"
    app_domain: str = "unknown"
    app_category: str = "unknown"
    device_id: str = "unknown"
    device_ip: str = "unknown"
    device_model: str = "unknown"
    device_type: int = 0
    device_make: str = "unknown"
    C1: int = 1005
    C14: int = 0
    C15: int = 320
    C16: int = 50
    C17: int = 0
    C18: int = 0
    C19: int = 0
    C20: int = 0
    C21: int = 0


class AdResponse(BaseModel):
    click_probability: float
    will_click: bool
    threshold: float = 0.5


@app.post("/predict", response_model=AdResponse)
async def predict(request: AdRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    from src.features.engineering import build_features, FEATURE_COLS

    df = pd.DataFrame([request.model_dump()])
    df, _ = build_features(df, encoders=_encoders)
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols]

    proba = float(_model.predict_proba(X)[0, 1])
    return AdResponse(click_probability=proba, will_click=proba >= 0.5)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/")
async def root():
    return {"service": "ad-click-prediction", "docs": "/docs"}
