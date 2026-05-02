"""Feature engineering for ad-click prediction (Avazu CTR dataset)."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# High-cardinality columns hashed to reduce dimensionality
HASH_COLS = ["site_id", "site_domain", "app_id", "app_domain", "device_id", "device_ip"]

# Low-cardinality categoricals encoded directly
LABEL_COLS = ["site_category", "app_category", "device_model", "device_type"]

# Numeric passthrough
NUMERIC_COLS = ["banner_pos", "C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hour-of-day and day-of-week from the Avazu `hour` column (format: YYMMDDHH)."""
    df = df.copy()
    hour_str = df["hour"].astype(str)
    df["hour_of_day"] = hour_str.str[-2:].astype(int)
    df["day_of_week"] = pd.to_datetime(hour_str.str[:6], format="%y%m%d").dt.dayofweek
    return df


def hash_high_cardinality(df: pd.DataFrame, n_buckets: int = 2**18) -> pd.DataFrame:
    """Hash high-cardinality ID columns into integer buckets."""
    df = df.copy()
    for col in HASH_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: hash(str(x)) % n_buckets)
    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Label-encode low-cardinality categoricals. Returns encoded df and fitted encoders."""
    df = df.copy()
    encoders = encoders or {}
    for col in LABEL_COLS:
        if col not in df.columns:
            continue
        if col not in encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
            )
    return df, encoders


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-feature interactions known to improve CTR models."""
    df = df.copy()
    # site × device type interaction
    if "site_id" in df.columns and "device_type" in df.columns:
        df["site_x_device"] = df["site_id"].astype(str) + "_" + df["device_type"].astype(str)
        df["site_x_device"] = df["site_x_device"].apply(lambda x: hash(x) % (2**18))
    # app × banner position interaction
    if "app_id" in df.columns and "banner_pos" in df.columns:
        df["app_x_banner"] = df["app_id"].astype(str) + "_" + df["banner_pos"].astype(str)
        df["app_x_banner"] = df["app_x_banner"].apply(lambda x: hash(x) % (2**18))
    return df


def build_features(df: pd.DataFrame, encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Full feature pipeline: time → hash → encode → interactions."""
    df = extract_time_features(df)
    df = hash_high_cardinality(df)
    df, encoders = encode_categoricals(df, encoders)
    df = add_interaction_features(df)
    return df, encoders


FEATURE_COLS = (
    NUMERIC_COLS
    + HASH_COLS
    + LABEL_COLS
    + ["hour_of_day", "day_of_week", "site_x_device", "app_x_banner"]
)
