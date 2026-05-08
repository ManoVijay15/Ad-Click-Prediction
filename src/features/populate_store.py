"""Populate the Redis feature store with per-entity historical CTR.

Run after training data is prepared:
    python -m src.features.populate_store [--sample 5_000_000]

Loads:
    feat:site:<site_id>      →  {ctr_7d, impr_7d}
    feat:app:<app_id>        →  {ctr_7d, impr_7d}
    feat:device:<device_id>  →  {ctr_7d, impr_7d}
"""

import argparse

import pandas as pd
from loguru import logger

from src.features.store import get_store


def populate(train_path: str, sample: int | None) -> None:
    logger.info(f"Loading {train_path}")
    df = pd.read_csv(train_path, dtype={"hour": str},
                     usecols=["click", "site_id", "app_id", "device_id"])
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
    logger.info(f"{len(df):,} rows loaded")

    store = get_store()
    store.bulk_load_ctr(df, entity_type="site", key_col="site_id")
    store.bulk_load_ctr(df, entity_type="app", key_col="app_id")
    store.bulk_load_ctr(df, entity_type="device", key_col="device_id")
    logger.info("Feature store populated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--sample", type=int, default=5_000_000,
                        help="Rows to sample (default 5M, pass 0 for full dataset)")
    args = parser.parse_args()
    populate(args.train, sample=args.sample if args.sample > 0 else None)
