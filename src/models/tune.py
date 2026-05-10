"""Hyperparameter tuning with Optuna + MLflow integration."""

import argparse
import os

import lightgbm as lgb
import mlflow
import optuna
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score

from src.features.engineering import FEATURE_COLS, build_features

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "ad-click-lgbm-tuning"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial, X_train, X_val, y_train, y_val) -> float:
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "n_estimators": 300,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(period=-1)],
        )
        val_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_proba)
        mlflow.log_metric("val_auc", auc)

    return auc


def tune(
    train_path: str = "data/processed/train.csv",
    val_path: str = "data/processed/val.csv",
    n_trials: int = 30,
    sample: int = 2_000_000,
) -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info(f"Loading train sample ({sample:,} rows)")
    train_df = pd.read_csv(train_path, dtype={"hour": str}).sample(n=sample, random_state=42)
    train_df, encoders = build_features(train_df)

    logger.info("Loading val sample (500k rows)")
    val_df = pd.read_csv(val_path, dtype={"hour": str}).sample(n=500_000, random_state=42)
    val_df, _ = build_features(val_df, encoders=encoders)

    feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    X_train = train_df[feature_cols]
    y_train = train_df["click"]
    X_val = val_df[feature_cols]
    y_val = val_df["click"]

    logger.info(f"Train: {len(X_train):,}  Val: {len(X_val):,}  CTR: {y_train.mean():.4f}")
    logger.info(f"Starting Optuna — {n_trials} trials")

    with mlflow.start_run(run_name="optuna-study"):
        study = optuna.create_study(direction="maximize", study_name="lgbm-ctr")
        study.optimize(
            lambda trial: objective(trial, X_train, X_val, y_train, y_val),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best = study.best_trial
        logger.info(f"Best AUC: {best.value:.4f}  Params: {best.params}")
        mlflow.log_metric("best_val_auc", best.value)
        mlflow.log_params(best.params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--val", default="data/processed/val.csv")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--sample", type=int, default=2_000_000)
    args = parser.parse_args()
    tune(args.train, args.val, n_trials=args.trials, sample=args.sample)
