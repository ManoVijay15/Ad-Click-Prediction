"""FastAPI serving layer for ad-click prediction."""

import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from src.features.engineering import build_features, FEATURE_COLS
from src.features.store import get_store
from src.models.loader import load_model_and_encoders

MODEL_NAME = os.getenv("MODEL_NAME", "ad-click-lgbm")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")

_state: dict = {"model": None, "encoders": None, "version": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, encoders, version = load_model_and_encoders(MODEL_NAME, MODEL_ALIAS)
    _state["model"] = model
    _state["encoders"] = encoders
    _state["version"] = version
    get_store()  # warm the feature store connection
    logger.info(f"API ready — {MODEL_NAME} v{version}")
    yield


app = FastAPI(title="Ad-Click Prediction API", version="0.1.0", lifespan=lifespan)


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
    model_version: str | None = None
    cached_features: dict[str, float] | None = None


@app.post("/predict", response_model=AdResponse)
async def predict(request: AdRequest):
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([request.model_dump()])
    df, _ = build_features(df, encoders=_state["encoders"])
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols]

    proba = float(model.predict_proba(X)[0, 1])

    store = get_store()
    cached = {
        **{f"site_{k}": v for k, v in store.get("site", request.site_id).items()},
        **{f"app_{k}": v for k, v in store.get("app", request.app_id).items()},
        **{f"device_{k}": v for k, v in store.get("device", request.device_id).items()},
    }

    return AdResponse(
        click_probability=proba,
        will_click=proba >= 0.5,
        model_version=_state["version"],
        cached_features=cached or None,
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "model_version": _state["version"],
    }


@app.get("/")
async def root():
    return {"service": "ad-click-prediction", "docs": "/docs"}
