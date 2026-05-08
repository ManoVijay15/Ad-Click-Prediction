"""Shared model + encoders loader for inference paths."""

import os
import pickle

import mlflow
import mlflow.lightgbm
from loguru import logger

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def load_model_and_encoders(model_name: str, alias: str = "production") -> tuple:
    """Load the registered model and its training-time encoders.

    Encoders are looked up by the `encoders_run_id` tag on the model version
    (set during training). Falls back to mv.run_id if the tag is missing.
    Returns (model, encoders_dict, version_str).
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    uri = f"models:/{model_name}@{alias}"
    logger.info(f"Loading {uri}")
    model = mlflow.lightgbm.load_model(uri)

    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(model_name, alias)
    run_id = mv.tags.get("encoders_run_id") or mv.run_id

    encoders: dict = {}
    if run_id:
        try:
            enc_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="encoders.pkl"
            )
            with open(enc_path, "rb") as f:
                encoders = pickle.load(f)
            logger.info(f"Encoders loaded from run {run_id[:8]}… ({len(encoders)} columns)")
        except Exception as e:
            logger.warning(f"Could not load encoders from run {run_id[:8]}: {e}")
    else:
        logger.warning("No run_id found for encoders — predictions will use uninitialized encoders")

    return model, encoders, mv.version
