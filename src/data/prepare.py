"""Avazu CTR dataset preparation.

Download the dataset from Kaggle before running:
    pip install kaggle
    kaggle competitions download -c avazu-ctr-prediction
    unzip avazu-ctr-prediction.zip -d data/raw/

Then run:
    python -m src.data.prepare [--sample 5_000_000]

Outputs:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv
    data/processed/stats.json
"""

import argparse
import json
import os
import pandas as pd
from loguru import logger

RAW_PATH = os.getenv("RAW_DATA_PATH", "data/raw/train.csv")
OUT_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")

# Avazu column dtypes — loading as string avoids mixed-type hash issues
DTYPE_MAP = {
    "id": str,
    "click": int,
    "hour": str,
    "C1": int,
    "banner_pos": int,
    "site_id": str,
    "site_domain": str,
    "site_category": str,
    "app_id": str,
    "app_domain": str,
    "app_category": str,
    "device_id": str,
    "device_ip": str,
    "device_model": str,
    "device_type": int,
    "device_conn_type": int,
    "device_make": str,
    "C14": int,
    "C15": int,
    "C16": int,
    "C17": int,
    "C18": int,
    "C19": int,
    "C20": int,
    "C21": int,
}


def load_raw(path: str, sample: int | None = None) -> pd.DataFrame:
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path, dtype=DTYPE_MAP)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
        logger.info(f"Sampled {len(df):,} rows")
    logger.info(f"Loaded {len(df):,} rows  CTR={df['click'].mean():.4f}")
    return df


def split_and_save(df: pd.DataFrame, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    # Chronological split on `hour` column to avoid data leakage
    df = df.sort_values("hour").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    train.to_csv(f"{out_dir}/train.csv", index=False)
    val.to_csv(f"{out_dir}/val.csv", index=False)
    test.to_csv(f"{out_dir}/test.csv", index=False)

    stats = {
        "total_rows": n,
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "overall_ctr": round(float(df["click"].mean()), 6),
        "train_ctr": round(float(train["click"].mean()), 6),
        "val_ctr": round(float(val["click"].mean()), 6),
        "test_ctr": round(float(test["click"].mean()), 6),
        "hour_min": str(df["hour"].min()),
        "hour_max": str(df["hour"].max()),
    }

    with open(f"{out_dir}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Splits saved → train={len(train):,}  val={len(val):,}  test={len(test):,}")
    logger.info(f"CTR — train={stats['train_ctr']:.4f}  val={stats['val_ctr']:.4f}  test={stats['test_ctr']:.4f}")
    return stats


def main(sample: int | None = None) -> None:
    df = load_raw(RAW_PATH, sample=sample)
    split_and_save(df, OUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Rows to sample (default: all)")
    args = parser.parse_args()
    main(sample=args.sample)
