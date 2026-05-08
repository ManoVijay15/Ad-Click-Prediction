"""Offline batch scoring — run a registered model over a CSV in chunks.

Run:
    python -m src.models.batch_score \
        --input  data/processed/test.csv \
        --output data/processed/test_predictions.parquet \
        --alias  production
"""

import argparse

import pandas as pd
from loguru import logger

from src.features.engineering import build_features, FEATURE_COLS
from src.models.loader import load_model_and_encoders

CHUNK_SIZE = 500_000


def score(input_path: str, output_path: str, model_name: str, alias: str) -> None:
    model, encoders, version = load_model_and_encoders(model_name, alias)
    logger.info(f"Using {model_name} v{version}")

    logger.info(f"Scoring {input_path} in chunks of {CHUNK_SIZE:,}")
    chunks = pd.read_csv(input_path, dtype={"hour": str}, chunksize=CHUNK_SIZE)

    results: list[pd.DataFrame] = []
    total = 0
    for i, chunk in enumerate(chunks, start=1):
        chunk_feat, _ = build_features(chunk, encoders=encoders)
        feature_cols = [c for c in FEATURE_COLS if c in chunk_feat.columns]
        proba = model.predict_proba(chunk_feat[feature_cols])[:, 1]

        out = pd.DataFrame({
            "id": chunk["id"] if "id" in chunk.columns else chunk.index,
            "click_probability": proba,
        })
        if "click" in chunk.columns:
            out["actual"] = chunk["click"].values

        results.append(out)
        total += len(chunk)
        logger.info(f"Chunk {i}: scored {total:,} rows  (CTR pred={proba.mean():.4f})")

    final = pd.concat(results, ignore_index=True)
    if output_path.endswith(".parquet"):
        final.to_parquet(output_path, index=False)
    else:
        final.to_csv(output_path, index=False)
    logger.info(f"Wrote {len(final):,} predictions → {output_path}")

    if "actual" in final.columns:
        from sklearn.metrics import roc_auc_score, log_loss
        auc = roc_auc_score(final["actual"], final["click_probability"])
        ll = log_loss(final["actual"], final["click_probability"])
        logger.info(f"Holdout AUC: {auc:.4f}  LogLoss: {ll:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="ad-click-lgbm")
    parser.add_argument("--alias", default="production")
    args = parser.parse_args()
    score(args.input, args.output, args.model, args.alias)
