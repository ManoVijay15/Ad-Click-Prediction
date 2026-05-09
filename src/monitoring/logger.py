"""PostgreSQL-backed prediction logger.

Records every /predict call so we can:
  - Compute live CTR vs predicted CTR
  - Detect drift over time
  - Update with ground-truth labels when feedback arrives
  - Power the Streamlit dashboard

Schema:
    predictions (
        id                BIGSERIAL PRIMARY KEY,
        ts                TIMESTAMPTZ DEFAULT NOW(),
        click_probability DOUBLE PRECISION,
        will_click        BOOLEAN,
        model_version     TEXT,
        request_payload   JSONB,
        cached_features   JSONB,
        actual_click      INT NULL          -- populated by feedback loop
    )
"""

from __future__ import annotations

import json
import os
from typing import Any

import psycopg2
import psycopg2.extras
from loguru import logger

PG_DSN = os.getenv(
    "PREDICTION_LOG_DSN",
    "postgresql://adclick:adclick@localhost:5432/adclick_mlflow",
)

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id                BIGSERIAL PRIMARY KEY,
    ts                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    click_probability DOUBLE PRECISION NOT NULL,
    will_click        BOOLEAN NOT NULL,
    model_version     TEXT,
    request_payload   JSONB,
    cached_features   JSONB,
    actual_click      INT
);
CREATE INDEX IF NOT EXISTS predictions_ts_idx ON predictions (ts DESC);
CREATE INDEX IF NOT EXISTS predictions_version_idx ON predictions (model_version);
"""

INSERT_SQL = """
INSERT INTO predictions
    (click_probability, will_click, model_version, request_payload, cached_features)
VALUES (%s, %s, %s, %s, %s);
"""


class PredictionLogger:
    """Singleton-style logger with graceful degradation when DB is unavailable."""

    def __init__(self, dsn: str = PG_DSN):
        self.dsn = dsn
        self._init()

    def _init(self) -> None:
        try:
            with psycopg2.connect(self.dsn) as conn, conn.cursor() as cur:
                cur.execute(CREATE_SQL)
            self.enabled = True
            logger.info(f"PredictionLogger ready at {self.dsn.split('@')[-1]}")
        except Exception as e:
            self.enabled = False
            logger.warning(f"PredictionLogger disabled (DB unreachable): {e}")

    def log(
        self,
        click_probability: float,
        will_click: bool,
        model_version: str | None,
        request_payload: dict[str, Any],
        cached_features: dict[str, float] | None,
    ) -> None:
        if not self.enabled:
            return
        try:
            with psycopg2.connect(self.dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    INSERT_SQL,
                    (
                        float(click_probability),
                        bool(will_click),
                        model_version,
                        json.dumps(request_payload),
                        json.dumps(cached_features) if cached_features else None,
                    ),
                )
        except Exception as e:
            logger.warning(f"PredictionLogger write failed: {e}")


_singleton: PredictionLogger | None = None


def get_logger() -> PredictionLogger:
    global _singleton
    if _singleton is None:
        _singleton = PredictionLogger()
    return _singleton
