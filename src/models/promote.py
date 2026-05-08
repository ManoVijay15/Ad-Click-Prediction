"""Promote a registered model version to a deployment alias.

MLflow 3.x deprecated stages (Production/Staging) in favor of aliases.
This script sets the @production alias on a model version so the API can
load it via `models:/ad-click-lgbm@production`.

Run:
    python -m src.models.promote                # promote latest version
    python -m src.models.promote --version 2    # promote a specific version
    python -m src.models.promote --alias staging
"""

import argparse
import os

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def promote(model_name: str, version: int | None, alias: str) -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    if version is None:
        # Try a few filter syntaxes — MLflow REST is picky about quoting
        versions = []
        for f in (f"name = '{model_name}'", f"name='{model_name}'"):
            versions = client.search_model_versions(f)
            if versions:
                break
        if not versions:
            try:
                rm = client.get_registered_model(model_name)
                versions = rm.latest_versions
            except Exception as e:
                raise RuntimeError(f"No versions found for model {model_name}: {e}")
        version = max(int(v.version) for v in versions)
        logger.info(f"Latest version of {model_name}: {version}")

    client.set_registered_model_alias(model_name, alias, str(version))
    logger.info(f"Set alias @{alias} → {model_name} v{version}")
    logger.info(f"Load via: models:/{model_name}@{alias}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ad-click-lgbm")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--alias", default="production")
    args = parser.parse_args()
    promote(args.model, args.version, args.alias)
