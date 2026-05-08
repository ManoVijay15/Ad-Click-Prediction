"""Train LightGBM CTR model and register it in MLflow."""

import argparse
import os
import pickle
import tempfile

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score, log_loss

from src.features.engineering import build_features, FEATURE_COLS

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "ad-click-lgbm"
MODEL_NAME = "ad-click-lgbm"
ENCODERS_ARTIFACT = "encoders.pkl"

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


def load_data(path: str, sample: int | None = None) -> pd.DataFrame:
    logger.info(f"Loading data from {path}" + (f" (sample={sample:,})" if sample else ""))
    df = pd.read_csv(path, dtype={"hour": str})
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
    return df


def train(
    train_path: str = "data/processed/train.csv",
    val_path: str = "data/processed/val.csv",
    register: bool = False,
    sample: int | None = 5_000_000,
) -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("Building train features")
    train_df = load_data(train_path, sample=sample)
    train_df, encoders = build_features(train_df)

    logger.info("Building val features")
    val_df = load_data(val_path, sample=500_000)
    val_df, _ = build_features(val_df, encoders=encoders)

    feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    X_train = train_df[feature_cols]
    y_train = train_df["click"]
    X_val = val_df[feature_cols]
    y_val = val_df["click"]

    logger.info(f"Train: {len(X_train):,}  Val: {len(X_val):,}  CTR: {y_train.mean():.4f}")

    with mlflow.start_run() as run:
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

        # Log encoders at run level — inference downloads via run_id
        with tempfile.TemporaryDirectory() as tmp:
            enc_path = f"{tmp}/{ENCODERS_ARTIFACT}"
            with open(enc_path, "wb") as f:
                pickle.dump(encoders, f)
            mlflow.log_artifact(enc_path)

        # Log the LightGBM model — MLflow 3.x creates a logged_model
        model_info = mlflow.lightgbm.log_model(model, name="model")
        logger.info(f"Logged model URI: {model_info.model_uri}")

        if register:
            mv = mlflow.register_model(model_info.model_uri, MODEL_NAME)
            # Stamp the version with the run_id holding the encoders
            client = mlflow.MlflowClient()
            client.set_model_version_tag(MODEL_NAME, mv.version, "encoders_run_id", run.info.run_id)
            logger.info(f"Registered {MODEL_NAME} v{mv.version} (encoders_run_id={run.info.run_id[:8]}…)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--val", default="data/processed/val.csv")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--sample", type=int, default=5_000_000,
                        help="Train rows to use (default 5M for dev; pass 0 for full dataset)")
    args = parser.parse_args()
    train(args.train, args.val, register=args.register,
          sample=args.sample if args.sample > 0 else None)
