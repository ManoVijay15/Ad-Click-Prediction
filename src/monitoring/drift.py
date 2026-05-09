"""Data drift detection comparing live predictions to training reference.

Run:
    python -m src.monitoring.drift \
        --reference data/processed/train.csv \
        --output    reports/drift.html
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import psycopg2
from loguru import logger

PG_DSN = os.getenv(
    "PREDICTION_LOG_DSN",
    "postgresql://adclick:adclick@localhost:5432/adclick_mlflow",
)

NUMERIC_COLS = ["banner_pos", "C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]
CATEGORICAL_COLS = [
    "site_category", "app_category", "device_type", "device_model",
    "site_id", "app_id",
]


def load_live_data(limit: int = 100_000) -> pd.DataFrame:
    """Read recent prediction request payloads from PostgreSQL."""
    logger.info(f"Loading up to {limit:,} live predictions from PostgreSQL")
    with psycopg2.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT request_payload, click_probability, actual_click "
            f"FROM predictions ORDER BY ts DESC LIMIT {limit}"
        )
        rows = cur.fetchall()

    if not rows:
        return pd.DataFrame()

    # JSONB columns come back as dicts already — no json.loads needed
    payloads = [r[0] if isinstance(r[0], dict) else json.loads(r[0]) for r in rows]
    df = pd.json_normalize(payloads)
    df["click_probability"] = [r[1] for r in rows]
    actuals = [r[2] for r in rows]
    if any(a is not None for a in actuals):
        df["actual_click"] = actuals
    return df


def detect_drift(
    reference_path: str,
    output_path: str = "reports/drift.html",
    sample: int = 100_000,
) -> dict:
    from evidently import Dataset, DataDefinition, Report
    from evidently.presets import DataDriftPreset

    logger.info(f"Loading reference from {reference_path}")
    reference = pd.read_csv(reference_path, dtype={"hour": str}, nrows=sample)

    current = load_live_data(limit=sample)
    if current.empty:
        logger.warning("No live predictions logged yet — drift report skipped")
        return {"drift_detected": False, "share_of_drifted_columns": 0.0, "report_path": None}

    common_num = [c for c in NUMERIC_COLS if c in reference.columns and c in current.columns]
    common_cat = [c for c in CATEGORICAL_COLS if c in reference.columns and c in current.columns]
    cols = common_num + common_cat
    reference, current = reference[cols], current[cols]

    definition = DataDefinition(numerical_columns=common_num, categorical_columns=common_cat)
    ref_ds = Dataset.from_pandas(reference, data_definition=definition)
    cur_ds = Dataset.from_pandas(current, data_definition=definition)

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(output_path)

    # The first metric in DataDriftPreset is the summary {count, share}
    drifted = 0
    share = 0.0
    try:
        summary = snapshot.dict().get("metrics", [{}])[0].get("value", {})
        if isinstance(summary, dict):
            drifted = int(summary.get("count", 0))
            share = float(summary.get("share", 0.0))
    except Exception as e:
        logger.warning(f"Could not parse drift summary ({e}) — see HTML report")

    logger.info(f"Drift columns: {drifted}/{len(cols)}  share={share:.1%}  report={output_path}")
    return {
        "drift_detected": share > 0.30,
        "share_of_drifted_columns": share,
        "report_path": output_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/processed/train.csv")
    parser.add_argument("--output", default="reports/drift.html")
    parser.add_argument("--sample", type=int, default=100_000)
    args = parser.parse_args()
    detect_drift(args.reference, args.output, args.sample)
