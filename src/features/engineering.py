"""Feature engineering for ad-click prediction (Avazu CTR dataset)."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# High-cardinality columns hashed to reduce dimensionality
HASH_COLS = ["site_id", "site_domain", "app_id", "app_domain", "device_id", "device_ip"]

# Low-cardinality categoricals encoded directly
LABEL_COLS = ["site_category", "app_category", "device_model", "device_type"]

# Numeric passthrough
NUMERIC_COLS = ["banner_pos", "C1", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]

N_BUCKETS = 2**18


def _vec_hash(series: pd.Series, n_buckets: int = N_BUCKETS) -> pd.Series:
    """Deterministic vectorized hash.

    Uses pandas' hash_pandas_object (xxhash-based, fixed seed) so the same
    input string maps to the same bucket across processes — critical for
    train/serve consistency.
    """
    hashes = pd.util.hash_pandas_object(series.astype(str), index=False)
    return (hashes % n_buckets).astype("int64")


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour_str = df["hour"].astype(str)
    df["hour_of_day"] = hour_str.str[-2:].astype(int)

    # Only ~177 unique date values in the dataset — convert once, then map
    date_str = hour_str.str[:6]
    dow_map = {
        h: pd.to_datetime(h, format="%y%m%d").dayofweek
        for h in date_str.unique()
    }
    df["day_of_week"] = date_str.map(dow_map)
    return df


def hash_high_cardinality(df: pd.DataFrame, n_buckets: int = N_BUCKETS) -> pd.DataFrame:
    df = df.copy()
    for col in HASH_COLS:
        if col in df.columns:
            df[col] = _vec_hash(df[col], n_buckets)
    return df


def encode_categoricals(
    df: pd.DataFrame, encoders: dict | None = None
) -> tuple[pd.DataFrame, dict]:
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
            # Vectorized dict map — unknown values → -1
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
    return df, encoders


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "site_id" in df.columns and "device_type" in df.columns:
        df["site_x_device"] = _vec_hash(
            df["site_id"].astype(str) + "_" + df["device_type"].astype(str)
        )
    if "app_id" in df.columns and "banner_pos" in df.columns:
        df["app_x_banner"] = _vec_hash(
            df["app_id"].astype(str) + "_" + df["banner_pos"].astype(str)
        )
    return df


def build_features(df: pd.DataFrame, encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
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
