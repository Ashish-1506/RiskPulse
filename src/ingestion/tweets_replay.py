"""Replay ingestion source for Kaggle stock tweets and synthetic social discourse."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd

from src.config import BASE_DIR, logger
from src.ingestion.base import Source
from src.ingestion.normalise import NormalizedDoc

DATE_CANDIDATES = ["timestamp", "date", "created_at", "datetime", "post_date", "time"]
TEXT_CANDIDATES = ["text", "tweet", "tweet_text", "body", "content", "message"]
TICKER_CANDIDATES = ["ticker", "symbol", "stock_symbol", "stock", "company"]


class TweetsReplaySource(Source):
    """Replays historical stock tweets from raw Kaggle CSV or synthetic fallback sample."""

    def __init__(self, file_path: Optional[str | Path] = None, speed: float = 0.0) -> None:
        """
        Args:
            file_path: Specific CSV path. If None, auto-discovers in data/raw/ or falls back.
            speed: Replay speed in rows/second. 0.0 means 'all at once' (no sleep delay).
        """
        self.file_path = self._resolve_data_file(file_path)
        self.speed = speed
        self._records: Optional[List[NormalizedDoc]] = None

    def _resolve_data_file(self, custom_path: Optional[str | Path]) -> Path:
        """Resolves existing CSV file from raw directory or fallback sample."""
        if custom_path:
            p = Path(custom_path)
            if p.is_file():
                return p

        # Check data/raw for any CSV
        raw_dir = BASE_DIR / "data" / "raw"
        if raw_dir.is_dir():
            csv_candidates = [f for f in raw_dir.glob("*.csv") if "cache" not in str(f)]
            if csv_candidates:
                logger.info(f"Using discovered raw tweets file: {csv_candidates[0]}")
                return csv_candidates[0]

        # Fallback to data/sample_tweets.csv
        sample_path = BASE_DIR / "data" / "sample_tweets.csv"
        if sample_path.is_file():
            logger.info("Using fallback sample tweets: data/sample_tweets.csv")
            return sample_path

        raise FileNotFoundError("No tweets CSV found in data/raw/ or data/sample_tweets.csv")

    def _detect_columns(self, df_cols: List[str]) -> tuple[str, str, Optional[str]]:
        """Auto-detects datetime, text, and optional ticker column names."""
        cols_lower = {col.lower(): col for col in df_cols}

        date_col = next((cols_lower[c] for c in DATE_CANDIDATES if c in cols_lower), None)
        text_col = next((cols_lower[c] for c in TEXT_CANDIDATES if c in cols_lower), None)
        ticker_col = next((cols_lower[c] for c in TICKER_CANDIDATES if c in cols_lower), None)

        if not date_col or not text_col:
            raise ValueError(f"Unable to auto-detect date and text columns in {df_cols}")

        return date_col, text_col, ticker_col

    def _load_and_normalize(self) -> List[NormalizedDoc]:
        """Loads CSV, auto-detects schema, normalizes and sorts by datetime."""
        if self._records is not None:
            return self._records

        logger.info(f"Loading tweets from {self.file_path}")
        df = pd.read_csv(self.file_path)
        date_col, text_col, ticker_col = self._detect_columns(list(df.columns))

        # Convert dates to UTC datetime
        df["_parsed_dt"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
        df = df.dropna(subset=["_parsed_dt", text_col])
        df = df.sort_values(by="_parsed_dt", ascending=True)

        docs: List[NormalizedDoc] = []
        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            if not text:
                continue

            dt = row["_parsed_dt"].to_pydatetime()
            ticker_val = str(row[ticker_col]).strip() if ticker_col and pd.notna(row.get(ticker_col)) else None

            # First few words serve as social headline
            title_snippet = (text[:80] + "...") if len(text) > 80 else text

            doc = NormalizedDoc.create(
                source="tweets",
                source_type="social",
                published_at=dt,
                title=title_snippet,
                text=text,
                url=None,
                language="en",
                ticker_hint=ticker_val,
            )
            docs.append(doc)

        self._records = docs
        logger.info(f"Successfully loaded and sorted {len(docs)} tweet documents.")
        return docs

    def fetch(self, since: Optional[datetime] = None, limit: int = 100) -> List[NormalizedDoc]:
        """Fetches normalized tweet records up to limit."""
        all_docs = self._load_and_normalize()
        if since:
            filtered = [d for d in all_docs if d.published_at >= since]
        else:
            filtered = all_docs

        return filtered[:limit]

    def stream(
        self,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> Iterator[NormalizedDoc]:
        """Streams tweets row by row with optional delay simulating live ingestion."""
        effective_speed = speed if speed is not None else self.speed
        docs = self.fetch(since=since, limit=limit or 10000)

        for doc in docs:
            if effective_speed > 0:
                time.sleep(1.0 / effective_speed)
            yield doc
