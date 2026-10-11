"""Disk caching utilities for raw ingestion API responses."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.config import BASE_DIR, logger

CACHE_DIR = BASE_DIR / "data" / "raw" / "cache"


def ensure_cache_dir() -> Path:
    """Ensures cache directory exists on disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def make_cache_key(prefix: str, payload: Any, hour_bucket: bool = True) -> str:
    """Generates a stable cache key string based on prefix, payload, and hour bucket."""
    raw_str = json.dumps(payload, sort_keys=True) if isinstance(payload, (dict, list)) else str(payload)
    if hour_bucket:
        now_hour = datetime.now(timezone.utc).strftime("%Y%m%d_%H")
        raw_str = f"{raw_str}_{now_hour}"
    digest = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}.json"


def get_cached_response(cache_key: str) -> Optional[Any]:
    """Retrieves cached JSON or text response from disk if present."""
    cache_path = CACHE_DIR / cache_key
    if not cache_path.is_file():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to read cache file {cache_key}: {exc}")
        return None


def set_cached_response(cache_key: str, data: Any) -> None:
    """Writes JSON serializable response to disk cache."""
    ensure_cache_dir()
    cache_path = CACHE_DIR / cache_key
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as exc:
        logger.warning(f"Failed to write cache file {cache_key}: {exc}")
