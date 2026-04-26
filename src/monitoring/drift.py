"""Data drift detection using Evidently AI."""

import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from loguru import logger


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, output_path: str = "reports/drift.html") -> dict:
    """Run drift detection between reference (train) and current (live) data.

    Returns a dict with drift detected flag and per-column drift scores.
    """
    column_mapping = ColumnMapping(target="click", prediction="click_probability")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
    report.save_html(output_path)

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    share_drifted = result["metrics"][0]["result"]["share_of_drifted_columns"]

    logger.info(f"Drift detected: {drift_detected}  Share drifted: {share_drifted:.2%}")
    return {
        "drift_detected": drift_detected,
        "share_of_drifted_columns": share_drifted,
        "report_path": output_path,
    }


def model_performance_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: str = "reports/performance.html",
) -> None:
    """Generate classification performance report."""
    column_mapping = ColumnMapping(target="click", prediction="click_probability")
    report = Report(metrics=[ClassificationPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
    report.save_html(output_path)
    logger.info(f"Performance report saved to {output_path}")
