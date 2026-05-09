"""FastAPI serving layer for ad-click prediction."""

import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from src.features.engineering import build_features, FEATURE_COLS
from src.features.store import get_store
from src.models.loader import load_model_and_encoders
from src.monitoring.logger import get_logger

MODEL_NAME = os.getenv("MODEL_NAME", "ad-click-lgbm")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")

_state: dict = {"model": None, "encoders": None, "version": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, encoders, version = load_model_and_encoders(MODEL_NAME, MODEL_ALIAS)
    _state["model"] = model
    _state["encoders"] = encoders
    _state["version"] = version
    get_store()
    get_logger()
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
async def predict(request: AdRequest, background: BackgroundTasks):
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    payload = request.model_dump()
    df = pd.DataFrame([payload])
    df, _ = build_features(df, encoders=_state["encoders"])
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    proba = float(model.predict_proba(df[feature_cols])[0, 1])
    will_click = proba >= 0.5

    store = get_store()
    cached = {
        **{f"site_{k}": v for k, v in store.get("site", request.site_id).items()},
        **{f"app_{k}": v for k, v in store.get("app", request.app_id).items()},
        **{f"device_{k}": v for k, v in store.get("device", request.device_id).items()},
    }

    background.add_task(
        get_logger().log,
        proba,
        will_click,
        _state["version"],
        payload,
        cached or None,
    )

    return AdResponse(
        click_probability=proba,
        will_click=will_click,
        model_version=_state["version"],
        cached_features=cached or None,
    )


class FeedbackRequest(BaseModel):
    prediction_id: int
    actual_click: int = Field(..., ge=0, le=1)


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    """Update a logged prediction with the ground-truth click label."""
    pred_logger = get_logger()
    if not pred_logger.enabled:
        raise HTTPException(status_code=503, detail="Prediction logger unavailable")
    import psycopg2
    try:
        with psycopg2.connect(pred_logger.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE predictions SET actual_click = %s WHERE id = %s",
                (request.actual_click, request.prediction_id),
            )
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="prediction_id not found")
        return {"status": "updated", "prediction_id": request.prediction_id}
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=str(e))


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
