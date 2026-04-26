"""Train LightGBM CTR model and register it in MLflow."""

import os
import pickle
import argparse
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from loguru import logger

from src.features.engineering import build_features, FEATURE_COLS

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "ad-click-lgbm"

LGBM_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "n_estimators": 500,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}


def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    return pd.read_csv(path)


def train(data_path: str, register: bool = False) -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(data_path)
    df, encoders = build_features(df)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols]
    y = df["click"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train):,}  Val: {len(X_val):,}  CTR: {y.mean():.4f}")

    with mlflow.start_run():
        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))
        mlflow.log_param("features", feature_cols)

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )

        val_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_proba)
        logloss = log_loss(y_val, val_proba)

        mlflow.log_metric("val_auc", auc)
        mlflow.log_metric("val_logloss", logloss)
        logger.info(f"Val AUC: {auc:.4f}  LogLoss: {logloss:.4f}")

        mlflow.lightgbm.log_model(model, artifact_path="model")

        # Save encoders alongside the model
        with open("encoders.pkl", "wb") as f:
            pickle.dump(encoders, f)
        mlflow.log_artifact("encoders.pkl")

        if register:
            run_id = mlflow.active_run().info.run_id
            mlflow.register_model(f"runs:/{run_id}/model", "ad-click-lgbm")
            logger.info("Model registered in MLflow Model Registry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/train.csv")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    train(args.data, register=args.register)
