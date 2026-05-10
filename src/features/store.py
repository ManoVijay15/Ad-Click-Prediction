"""Redis-backed feature store for ad-click prediction.

Provides a thin abstraction for storing and retrieving precomputed
per-entity features (e.g. historical CTR by site_id, app_id, device_id).
Falls back to default values if Redis is unavailable.

Layout in Redis:
    feat:<entity_type>:<entity_id>  →  hash {feature_name: value}

Example:
    feat:site:abc123      →  {ctr_7d: 0.184, impr_7d: 12500}
    feat:app:com.foo      →  {ctr_7d: 0.092}
    feat:device:hash99    →  {ctr_7d: 0.054, last_seen: 1714560000}
"""

from __future__ import annotations

import json
import os
from typing import Any

import redis
from loguru import logger

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DEFAULT_TTL = 7 * 24 * 3600  # 7 days


class FeatureStore:
    """Lightweight Redis feature store with graceful degradation."""

    def __init__(self, url: str = REDIS_URL, ttl: int = DEFAULT_TTL):
        self.ttl = ttl
        try:
            self.client: redis.Redis | None = redis.from_url(url, decode_responses=True)
            self.client.ping()
            logger.info(f"FeatureStore connected to {url}")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"FeatureStore offline ({e}) — get/set will return defaults")
            self.client = None

    @staticmethod
    def _key(entity_type: str, entity_id: str) -> str:
        return f"feat:{entity_type}:{entity_id}"

    def get(self, entity_type: str, entity_id: str) -> dict[str, float]:
        if self.client is None:
            return {}
        try:
            raw = self.client.hgetall(self._key(entity_type, entity_id))
            return {k: float(v) for k, v in raw.items()}
        except (redis.ConnectionError, redis.TimeoutError):
            return {}

    def set(self, entity_type: str, entity_id: str, features: dict[str, Any]) -> None:
        if self.client is None or not features:
            return
        try:
            key = self._key(entity_type, entity_id)
            pipe = self.client.pipeline()
            pipe.hset(key, mapping={
                k: json.dumps(v) if isinstance(v, dict) else v
                for k, v in features.items()
            })
            pipe.expire(key, self.ttl)
            pipe.execute()
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"FeatureStore set failed: {e}")

    def bulk_load_ctr(self, df, entity_type: str, key_col: str, label_col: str = "click") -> int:
        """Compute per-entity CTR + impressions and bulk-load into Redis.

        Returns number of entities loaded.
        """
        if self.client is None:
            logger.warning("FeatureStore offline — skipping bulk_load")
            return 0

        agg = df.groupby(key_col)[label_col].agg(["mean", "count"]).reset_index()
        agg.columns = [key_col, "ctr_7d", "impr_7d"]

        pipe = self.client.pipeline()
        for _, row in agg.iterrows():
            key = self._key(entity_type, str(row[key_col]))
            pipe.hset(key, mapping={"ctr_7d": float(row["ctr_7d"]), "impr_7d": int(row["impr_7d"])})
            pipe.expire(key, self.ttl)
        pipe.execute()
        logger.info(f"Loaded {len(agg):,} {entity_type} features into Redis")
        return len(agg)


_singleton: FeatureStore | None = None


def get_store() -> FeatureStore:
    global _singleton
    if _singleton is None:
        _singleton = FeatureStore()
    return _singleton
