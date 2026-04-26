"""Hyperparameter tuning with Optuna + MLflow integration."""

import os
import pandas as pd
import lightgbm as lgb
import optuna
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from loguru import logger

from src.features.engineering import build_features, FEATURE_COLS

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "ad-click-lgbm-tuning"


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
        "n_estimators": 500,
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


def tune(data_path: str = "data/processed/train.csv", n_trials: int = 30) -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(data_path)
    df, _ = build_features(df)
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols]
    y = df["click"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    with mlflow.start_run(run_name="optuna-study"):
        study = optuna.create_study(direction="maximize", study_name="lgbm-ctr")
        study.optimize(
            lambda trial: objective(trial, X_train, X_val, y_train, y_val),
            n_trials=n_trials,
        )

        best = study.best_trial
        logger.info(f"Best AUC: {best.value:.4f}  Params: {best.params}")
        mlflow.log_metric("best_val_auc", best.value)
        mlflow.log_params(best.params)


if __name__ == "__main__":
    tune()
