"""Logistic Regression baseline for ad-click CTR prediction.

Uses SGDClassifier(loss='log_loss') — mathematically identical to logistic
regression but scales to 28M rows via mini-batch training.

Run:
    python -m src.models.baseline [--train data/processed/train.csv]
                                  [--val   data/processed/val.csv]
                                  [--sample 3_000_000]
"""

import argparse
import os
import pickle
import tempfile

import mlflow
import pandas as pd
from loguru import logger
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import MaxAbsScaler

from src.features.engineering import FEATURE_COLS, build_features

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "ad-click-baseline"

CHUNK_SIZE = 200_000  # rows per mini-batch for partial_fit


def load_sample(path: str, n: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"hour": str})
    if n and n < len(df):
        df = df.sample(n=n, random_state=42).reset_index(drop=True)
    return df


def train_baseline(
    train_path: str = "data/processed/train.csv",
    val_path: str = "data/processed/val.csv",
    sample: int | None = 3_000_000,
) -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("Loading training data")
    train_df = load_sample(train_path, sample)
    train_df, encoders = build_features(train_df)

    logger.info("Loading validation data (500k sample)")
    val_df = pd.read_csv(val_path, dtype={"hour": str}).sample(500_000, random_state=42)
    val_df, _ = build_features(val_df, encoders=encoders)

    feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    X_train = train_df[feature_cols].values
    y_train = train_df["click"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["click"].values

    logger.info(f"Train: {len(X_train):,}  Val: {len(X_val):,}  CTR: {y_train.mean():.4f}")

    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    params = {
        "loss": "log_loss",
        "penalty": "l2",
        "alpha": 1e-4,
        "max_iter": 5,
        "random_state": 42,
        "n_jobs": -1,
        "sample": sample or "all",
    }

    with mlflow.start_run(run_name="logistic-regression-baseline"):
        mlflow.log_params(params)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))
        mlflow.log_param("features", feature_cols)

        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=params["alpha"],
            max_iter=1,
            warm_start=True,
            random_state=42,
            n_jobs=-1,
        )

        # Mini-batch partial_fit over multiple passes
        n_chunks = max(1, len(X_train) // CHUNK_SIZE)
        for epoch in range(params["max_iter"]):
            for i in range(n_chunks):
                start, end = i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE
                model.partial_fit(X_train[start:end], y_train[start:end], classes=[0, 1])

        val_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_proba)
        logloss = log_loss(y_val, val_proba)

        mlflow.log_metric("val_auc", auc)
        mlflow.log_metric("val_logloss", logloss)
        logger.info(f"Baseline — Val AUC: {auc:.4f}  LogLoss: {logloss:.4f}")

        with tempfile.TemporaryDirectory() as tmp:
            model_path = f"{tmp}/model.pkl"
            scaler_path = f"{tmp}/scaler.pkl"
            enc_path = f"{tmp}/encoders.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            with open(enc_path, "wb") as f:
                pickle.dump(encoders, f)
            mlflow.log_artifact(model_path, artifact_path="model")
            mlflow.log_artifact(scaler_path, artifact_path="model")
            mlflow.log_artifact(enc_path, artifact_path="model")

    logger.info("Baseline training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--val", default="data/processed/val.csv")
    parser.add_argument("--sample", type=int, default=3_000_000)
    args = parser.parse_args()
    train_baseline(args.train, args.val, args.sample)
